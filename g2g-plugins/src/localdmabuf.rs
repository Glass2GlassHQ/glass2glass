// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Vendor-neutral local zero-copy transport (`local-dmabuf`): the DMABUF
//! analog of the CUDA [`LocalCudaSink`](crate::localcuda::LocalCudaSink) /
//! [`LocalCudaSrc`](crate::localcuda::LocalCudaSrc) pair. `DmaBufSink`
//! (Unix-socket client) and `DmaBufSrc` (Unix-socket server) carry a
//! [`MemoryDomain::DmaBuf`] frame from one process to another *on the same
//! machine* with no copy, by passing the frame's dma-buf file descriptor over the
//! socket as `SCM_RIGHTS` ancillary data (see [`crate::scmfd`]).
//!
//! # Buffer lifetime
//!
//! `SCM_RIGHTS` makes the kernel install a *dup* of the fd in the receiver, and
//! the underlying memory is refcounted across both fds, so it outlives the
//! sink's frame. A pooled producer (a V4L2 decoder, a capture device) still
//! rewrites the buffer once the frame drops in its own process, so the sink keeps
//! one share of every frame it sent until the source answers with a RELEASE
//! record, sent when downstream drops the frame. Backpressure is still provided
//! by the graph's bounded channel upstream of the sink.
//!
//! # Contract
//!
//! - **Linux only** (dma-buf + `SCM_RIGHTS`); LP64 (see [`crate::scmfd`]).
//! - **GPU-agnostic.** The transport moves *any* dma-buf: a GPU-exported texture,
//!   a V4L2 / CSI capture buffer, a `dma_heap` / `udmabuf` allocation. Importing
//!   the received fd into a wgpu buffer is the separate `dmabuf-wgpu`
//!   ([`crate::dmabufwgpu`]) element on the receive side.
//! - The sink assumes `frame.domain` is `DmaBuf`; a system-memory frame belongs
//!   on the CPU wire codec (`remote` / `remote-ws`) instead.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::os::fd::AsRawFd;
use std::os::unix::net::UnixListener as StdUnixListener;
use std::sync::{Mutex, PoisonError};

use tokio::io::Interest;
use tokio::net::UnixStream;

use g2g_core::clock::ClockSync;
use g2g_core::log::io_err;
use g2g_core::memory::{DmaBufReleaseWaker, MemoryDomain, MemoryDomainKind, OwnedDmaBuf, SyncFd};
use g2g_core::meta::{Plane, PlaneLayout, MAX_PLANES};
use g2g_core::pad_template::{PadTemplate, PadTemplates};
use g2g_core::query::LatencyReport;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    raw_format_from_u8, raw_format_to_u8, AsyncElement, Caps, CapsConstraint, CapsSet,
    ConfigureOutcome, Dim, ElementMetadata, Frame, FrameTiming, G2gError, HardwareError,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};

/// The `location` property shared by both ends: the Unix socket path.
const LOCATION_SPEC: PropertySpec =
    PropertySpec::new("location", PropKind::Str, "Unix socket path");
const LOCATION_PROP: &[PropertySpec] = &[LOCATION_SPEC];

/// The receive end also takes gst `basesrc`'s frame count.
const SRC_PROPS: &[PropertySpec] = &[
    LOCATION_SPEC,
    crate::numbuffers::num_buffers_property(
        "frames to emit then EOS (-1 = until the sender ends the stream)",
    ),
];

use crate::abortondrop::AbortOnDrop;
use crate::arrivalstamp::SenderTimelineShift;
use crate::scmfd;

// ---- fixed-size wire records over the Unix socket ----
//
// Every message is one fixed-length record, sent / received with a single
// sendmsg / recvmsg so an accompanying fd (on a FRAME record) is never separated
// from its bytes. Fixed size means the receiver never over-reads across a record
// boundary, which on a stream socket would discard a pending fd.

const TAG_CAPS: u8 = 0;
const TAG_FRAME: u8 = 1;
const TAG_EOS: u8 = 2;
/// Carries the stream's exported timeline-semaphore fd (as ancillary data) once,
/// before the first synced frame. See the sync notes on [`DmaBufSink`].
const TAG_SYNC: u8 = 3;
// source to sink: downstream dropped the frame record at position `sequence`, counted from zero
const TAG_RELEASE: u8 = 4;

/// Fixed record length: tag + format + geometry + sequence + timing + keyframe +
/// sync value + plane layout.
const RECORD_LEN: usize = 1  // tag
    + 1                      // format (raw_format_to_u8)
    + 4 * 4                  // width, height, stride, offset
    + 8                      // sequence
    + 8 * 5                  // timing: pts, dts, dur, capture, arrival
    + 1                      // keyframe
    + 8                      // sync_value (timeline value to wait on; 0 = frame carries no sync)
    + 1                      // plane count (0 = the dma-buf carries no plane layout)
    + MAX_PLANES * PLANE_BYTES; // each plane's offset and stride
const PLANE_COUNT_AT: usize = 75;
const PLANES_AT: usize = PLANE_COUNT_AT + 1;
const PLANE_BYTES: usize = 8 * 2;

/// The per-frame descriptor (everything except the fd, which travels as
/// ancillary data). For a CAPS record only tag/format/width/height are read; for
/// EOS / SYNC only the tag (SYNC also carries an fd).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Record {
    tag: u8,
    format: u8,
    width: u32,
    height: u32,
    stride: u32,
    offset: u32,
    sequence: u64,
    timing: FrameTiming,
    /// Timeline value the consumer must host-wait before reading this frame's
    /// buffer (0 = no GPU sync; the producer synchronised itself).
    sync_value: u64,
    plane_layout: Option<PlaneLayout>,
}

impl Record {
    fn zeroed(tag: u8) -> Self {
        Self {
            tag,
            format: 0,
            width: 0,
            height: 0,
            stride: 0,
            offset: 0,
            sequence: 0,
            timing: FrameTiming::default(),
            sync_value: 0,
            plane_layout: None,
        }
    }

    fn encode(&self) -> [u8; RECORD_LEN] {
        let mut b = [0u8; RECORD_LEN];
        b[0] = self.tag;
        b[1] = self.format;
        b[2..6].copy_from_slice(&self.width.to_le_bytes());
        b[6..10].copy_from_slice(&self.height.to_le_bytes());
        b[10..14].copy_from_slice(&self.stride.to_le_bytes());
        b[14..18].copy_from_slice(&self.offset.to_le_bytes());
        b[18..26].copy_from_slice(&self.sequence.to_le_bytes());
        b[26..34].copy_from_slice(&self.timing.pts_ns.to_le_bytes());
        b[34..42].copy_from_slice(&self.timing.dts_ns.to_le_bytes());
        b[42..50].copy_from_slice(&self.timing.duration_ns.to_le_bytes());
        b[50..58].copy_from_slice(&self.timing.capture_ns.to_le_bytes());
        b[58..66].copy_from_slice(&self.timing.arrival_ns.to_le_bytes());
        b[66] = self.timing.keyframe as u8;
        b[67..75].copy_from_slice(&self.sync_value.to_le_bytes());
        if let Some(layout) = &self.plane_layout {
            b[PLANE_COUNT_AT] = layout.count() as u8;
            for (index, plane) in (0..layout.count())
                .filter_map(|index| layout.plane(index))
                .enumerate()
            {
                let at = PLANES_AT + index * PLANE_BYTES;
                b[at..at + 8].copy_from_slice(&(plane.offset as u64).to_le_bytes());
                b[at + 8..at + 16].copy_from_slice(&(plane.stride as u64).to_le_bytes());
            }
        }
        b
    }

    fn decode(b: &[u8; RECORD_LEN]) -> Option<Self> {
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        let plane_layout = match usize::from(b[PLANE_COUNT_AT]) {
            0 => None,
            count if count > MAX_PLANES => return None,
            count => {
                let planes = (0..count)
                    .map(|index| {
                        let at = PLANES_AT + index * PLANE_BYTES;
                        Some(Plane {
                            offset: usize::try_from(u64_at(at)).ok()?,
                            stride: usize::try_from(u64_at(at + 8)).ok()?,
                        })
                    })
                    .collect::<Option<Vec<_>>>()?;
                Some(PlaneLayout::new(&planes)?)
            }
        };
        Some(Self {
            tag: b[0],
            format: b[1],
            width: u32_at(2),
            height: u32_at(6),
            stride: u32_at(10),
            offset: u32_at(14),
            sequence: u64_at(18),
            timing: FrameTiming {
                pts_ns: u64_at(26),
                dts_ns: u64_at(34),
                duration_ns: u64_at(42),
                capture_ns: u64_at(50),
                arrival_ns: u64_at(58),
                keyframe: b[66] != 0,
                ..FrameTiming::default()
            },
            sync_value: u64_at(67),
            plane_layout,
        })
    }
}

/// Send one record, attaching `fd` (a FRAME's dma-buf) as `SCM_RIGHTS`. The fd
/// rides the first `sendmsg`; a short write sends the remainder without it.
async fn send_record(sock: &UnixStream, rec: &Record, fd: Option<i32>) -> Result<(), G2gError> {
    let buf = rec.encode();
    let raw = sock.as_raw_fd();
    let mut sent = 0usize;
    let mut fd = fd;
    while sent < buf.len() {
        sock.writable().await.map_err(io_err)?;
        match sock.try_io(Interest::WRITABLE, || {
            scmfd::send_with_fd(raw, &buf[sent..], fd)
        }) {
            Ok(n) => {
                sent += n;
                fd = None; // ancillary fd only accompanies the first chunk
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
            Err(e) => return Err(io_err(e)),
        }
    }
    Ok(())
}

/// Read one full record, capturing an fd if one accompanies it. `Ok(None)` on a
/// clean EOF at a record boundary.
async fn recv_record(sock: &UnixStream) -> Result<Option<(Record, Option<i32>)>, G2gError> {
    let raw = sock.as_raw_fd();
    let mut buf = [0u8; RECORD_LEN];
    let mut got = 0usize;
    let mut fd: Option<i32> = None;
    while got < buf.len() {
        sock.readable().await.map_err(io_err)?;
        let (n, f) = match sock.try_io(Interest::READABLE, || {
            scmfd::recv_with_fd(raw, &mut buf[got..])
        }) {
            Ok(v) => v,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
            Err(e) => return Err(io_err(e)),
        };
        if n == 0 {
            if got == 0 {
                return Ok(None); // clean EOF between records
            }
            return Err(G2gError::Hardware(HardwareError::Other)); // truncated record
        }
        if let Some(rfd) = f {
            // A record carries at most one fd; a second is a protocol error.
            if fd.is_some() {
                close_fd(rfd);
                return Err(G2gError::Hardware(HardwareError::Other));
            }
            fd = Some(rfd);
        }
        got += n;
    }
    let Some(record) = Record::decode(&buf) else {
        if let Some(fd) = fd {
            close_fd(fd);
        }
        return Err(G2gError::Hardware(HardwareError::Other));
    };
    Ok(Some((record, fd)))
}

async fn apply_releases(socket: Arc<UnixStream>, lent: Arc<Mutex<Vec<(u64, OwnedDmaBuf)>>>) {
    while let Ok(Some((record, fd))) = recv_record(&socket).await {
        if let Some(fd) = fd {
            close_fd(fd);
        }
        if record.tag == TAG_RELEASE {
            lent.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .retain(|(position, _)| *position != record.sequence);
        }
    }
    lent.lock().unwrap_or_else(PoisonError::into_inner).clear();
}

/// Close a received fd we will not keep (error paths). Best-effort.
fn close_fd(fd: i32) {
    extern "C" {
        fn close(fd: i32) -> i32;
    }
    // SAFETY: `fd` was just received via SCM_RIGHTS (owned by this process) and is
    // closed exactly once here on an error path.
    unsafe {
        close(fd);
    }
}

/// Raw video formats the transport carries (matches the `dmabuf-wgpu` import
/// side; the pixel caps pass through, only the transport carries them).
const FORMATS: [RawVideoFormat; 4] = [
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
    RawVideoFormat::Nv12,
    RawVideoFormat::I420,
];

fn any_dmabuf_caps() -> CapsSet {
    let any = |format| Caps::RawVideo {
        format,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
    };
    CapsSet::from_alternatives(FORMATS.map(any).to_vec())
}

fn caps_of(format: RawVideoFormat, width: u32, height: u32) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        // A source must advertise a fixable rate: negotiation's fixate() rejects
        // Rate::Any. Per-frame timing crosses in the record; this nominal rate
        // only satisfies fixation (the transport is rate-agnostic).
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
    }
}

fn dims_of(caps: &Caps) -> Option<(RawVideoFormat, u32, u32)> {
    match caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } => Some((*format, *w, *h)),
        _ => None,
    }
}

// ---- sink (client) ----

/// # Example
///
/// ```no_run
/// use g2g_plugins::localdmabuf::DmaBufSink;
///
/// let element = DmaBufSink::new("/tmp/g2g-dmabuf.sock");
/// ```
#[derive(Debug)]
pub struct DmaBufSink {
    path: String,
    socket: Option<Arc<UnixStream>>,
    // one share of every frame the source has not released, so a pooled producer cannot reuse its buffer
    lent: Arc<Mutex<Vec<(u64, OwnedDmaBuf)>>>,
    releases: Option<AbortOnDrop>,
    configured_caps: Option<Caps>,
    caps_sent: bool,
    /// Whether the stream's timeline-semaphore fd has been shared (once, before the
    /// first synced frame). See the sync notes below.
    sync_sent: bool,
    configured: bool,
    sent: u64,
}

impl DmaBufSink {
    /// Send dma-buf frames to `path` (the [`DmaBufSrc`] Unix socket).
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            socket: None,
            lent: Arc::default(),
            releases: None,
            configured_caps: None,
            caps_sent: false,
            sync_sent: false,
            configured: false,
            sent: 0,
        }
    }

    /// Frames handed to the peer. Useful in tests.
    pub fn sent(&self) -> u64 {
        self.sent
    }
}

impl PadTemplates for DmaBufSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(any_dmabuf_caps())])
    }
}

impl AsyncElement for DmaBufSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Pass through the pixel caps unchanged (only the memory domain matters).
        if dims_of(upstream_caps).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(any_dmabuf_caps())
    }

    /// Accepts only dma-buf frames (domain negotiation), so the auto-plug splices
    /// this where an upstream produces a dma-buf.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(MemoryDomainKind::DmaBuf)
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
                let socket = Arc::new(UnixStream::connect(&self.path).await.map_err(io_err)?);
                let releases =
                    tokio::spawn(apply_releases(Arc::clone(&socket), Arc::clone(&self.lent)));
                self.releases = Some(AbortOnDrop(releases));
                self.socket = Some(socket);
            }
            // Announce format + dims once so the receiver discovers its caps.
            if !self.caps_sent {
                let (format, w, h) = self
                    .configured_caps
                    .as_ref()
                    .and_then(dims_of)
                    .ok_or(G2gError::CapsMismatch)?;
                let mut rec = Record::zeroed(TAG_CAPS);
                rec.format = raw_format_to_u8(format);
                rec.width = w;
                rec.height = h;
                let sock = self.socket.as_ref().unwrap();
                send_record(sock, &rec, None).await?;
                self.caps_sent = true;
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::DmaBuf(dmabuf) = &frame.domain else {
                        // Only dma-buf frames pass by fd; a CPU frame belongs on
                        // the wire codec instead.
                        return Err(G2gError::UnsupportedDomain);
                    };
                    dmabuf.require_linear(g2g_core::log::short_type_name::<Self>())?;
                    let (format, w, h) = self
                        .configured_caps
                        .as_ref()
                        .and_then(dims_of)
                        .ok_or(G2gError::CapsMismatch)?;
                    // One-time: if the producer attached a GPU-completion timeline
                    // semaphore (zero-stall `WgpuToDmaBuf`), share its fd once so the
                    // peer can host-wait each frame's `sync_value`. The semaphore is
                    // one per stream, so it need travel only once.
                    if !self.sync_sent {
                        if let Some(sync_fd) = dmabuf.sync_fd() {
                            let sock = self.socket.as_ref().unwrap();
                            send_record(sock, &Record::zeroed(TAG_SYNC), Some(sync_fd)).await?;
                            self.sync_sent = true;
                        }
                    }
                    let rec = Record {
                        tag: TAG_FRAME,
                        format: raw_format_to_u8(format),
                        width: w,
                        height: h,
                        stride: dmabuf.stride,
                        offset: dmabuf.offset,
                        sequence: frame.sequence,
                        timing: frame.timing,
                        sync_value: dmabuf.sync_value().unwrap_or(0),
                        plane_layout: dmabuf.plane_layout().copied(),
                    };
                    let fd = dmabuf.as_raw();
                    // a pooled producer such as a V4L2 decoder rewrites the buffer once every share in this process drops
                    self.lent
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .push((self.sent, dmabuf.clone()));
                    let sock = self.socket.as_ref().unwrap();
                    send_record(sock, &rec, Some(fd)).await?;
                    self.sent += 1;
                }
                PipelinePacket::Eos => {
                    // the release task keeps the socket open until the source closes it
                    if let Some(sock) = self.socket.take() {
                        let _ = send_record(&sock, &Record::zeroed(TAG_EOS), None).await;
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
            "Local DMABUF sink",
            "Sink/Network",
            "Shares dma-buf frames with a same-machine peer over a Unix socket via SCM_RIGHTS fd passing (no copy, vendor-neutral)",
            crate::AARON_BOXER,
        )
    }
}

// ---- source (server) ----

/// # Example
///
/// ```no_run
/// use g2g_plugins::localdmabuf::DmaBufSrc;
///
/// let element = DmaBufSrc::new("/tmp/g2g-dmabuf.sock").with_frame_limit(300);
/// ```
#[derive(Debug)]
pub struct DmaBufSrc {
    path: String,
    listener: Option<StdUnixListener>,
    socket: Option<UnixStream>,
    discovered: Option<Caps>,
    configured: bool,
    /// `u64::MAX` runs until the sender ends the stream; otherwise stop after
    /// this many frames and emit EOS.
    frame_limit: u64,
    clock_sync: Option<ClockSync>,
}

impl DmaBufSrc {
    /// Receive dma-buf frames on the Unix socket `path` (the [`DmaBufSink`]
    /// connects here). Any stale socket file at `path` is removed on bind.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            listener: None,
            socket: None,
            discovered: None,
            configured: false,
            frame_limit: u64::MAX,
            clock_sync: None,
        }
    }

    /// Stop after `n` frames and emit EOS (the bounded / test path). 0 emits EOS
    /// without reading from the sender.
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    /// Accept the sender and read its first record (the caps). Idempotent.
    async fn ensure_connected(&mut self) -> Result<Caps, G2gError> {
        if let Some(caps) = &self.discovered {
            return Ok(caps.clone());
        }
        let listener = match self.listener.take() {
            Some(l) => l,
            None => {
                let _ = std::fs::remove_file(&self.path);
                StdUnixListener::bind(&self.path).map_err(io_err)?
            }
        };
        listener.set_nonblocking(true).map_err(io_err)?;
        let listener = tokio::net::UnixListener::from_std(listener).map_err(io_err)?;
        let (stream, _) = listener.accept().await.map_err(io_err)?;
        let (rec, fd) = recv_record(&stream).await?.ok_or(G2gError::NotConfigured)?;
        if let Some(fd) = fd {
            close_fd(fd); // a caps record must not carry an fd
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        if rec.tag != TAG_CAPS {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        let format = raw_format_from_u8(rec.format).map_err(|_| G2gError::CapsMismatch)?;
        let caps = caps_of(format, rec.width, rec.height);
        self.socket = Some(stream);
        self.discovered = Some(caps.clone());
        Ok(caps)
    }
}

impl SourceLoop for DmaBufSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = Pin<Box<dyn Future<Output = Result<Caps, G2gError>> + 'a>>
    where
        Self: 'a;

    fn latency(&self) -> LatencyReport {
        LatencyReport::live(0, None)
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.clock_sync = Some(sync);
    }

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
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::DmaBuf
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.path = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.frame_limit, &value),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.path.clone())),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.frame_limit)),
            _ => None,
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Local DMABUF source",
            "Source/Network",
            "Receives dma-buf frames from a same-machine DmaBufSink over a Unix socket via SCM_RIGHTS fd passing",
            crate::AARON_BOXER,
        )
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            if crate::numbuffers::finished_at_zero_limit(self.frame_limit, out).await? {
                return Ok(0);
            }
            let caps = self.discovered.clone().ok_or(G2gError::NotConfigured)?;
            let sock = self.socket.take().ok_or(G2gError::NotConfigured)?;
            out.push(PipelinePacket::CapsChanged(caps)).await?;

            let limit = self.frame_limit;
            let mut emitted = 0u64;
            let mut timeline = SenderTimelineShift::new(self.clock_sync.clone());
            // The stream's GPU-completion timeline semaphore, received once (TAG_SYNC)
            // and shared (refcounted) into every synced frame; a sem-aware consumer
            // (`DmaBufToWgpu`) host-waits each frame's value before reading.
            let mut sync: Option<SyncFd> = None;
            let release_waker = Arc::new(DmaBufReleaseWaker::default());
            // one share of every frame downstream still holds, by its position in the stream
            let mut lent: Vec<(u64, OwnedDmaBuf)> = Vec::new();
            let mut frames_received = 0u64;
            // a dropped `recv_record` loses the part of a record it already read
            let mut receiving = Box::pin(recv_record(&sock));
            loop {
                let received = core::future::poll_fn(|cx| {
                    release_waker.register(cx.waker());
                    if let Some(index) = lent.iter().position(|(_, lent)| lent.share_count() == 1) {
                        return core::task::Poll::Ready(Err(index));
                    }
                    receiving.as_mut().poll(cx).map(Ok)
                })
                .await;
                let received = match received {
                    Err(index) => {
                        let (position, _) = lent.swap_remove(index);
                        let mut release = Record::zeroed(TAG_RELEASE);
                        release.sequence = position;
                        // a sink that exited after its EOS record still has that record waiting to be read
                        let _ = send_record(&sock, &release, None).await;
                        continue;
                    }
                    Ok(received) => {
                        receiving = Box::pin(recv_record(&sock));
                        received
                    }
                };
                let (rec, fd) = match received? {
                    Some(m) => m,
                    None => {
                        out.push(PipelinePacket::Eos).await?;
                        break;
                    }
                };
                match rec.tag {
                    TAG_SYNC => {
                        match fd {
                            // SAFETY: `fd` was just received via SCM_RIGHTS, so this
                            // process owns it; `SyncFd` closes it once on last drop.
                            Some(fd) => sync = Some(unsafe { SyncFd::from_raw(fd) }),
                            None => return Err(G2gError::Hardware(HardwareError::Other)),
                        }
                    }
                    TAG_EOS => {
                        if let Some(fd) = fd {
                            close_fd(fd);
                        }
                        out.push(PipelinePacket::Eos).await?;
                        break;
                    }
                    TAG_FRAME => {
                        let Some(fd) = fd else {
                            // A frame record without its fd: the kernel dropped the
                            // ancillary data (protocol / peer error). Fail loudly.
                            return Err(G2gError::Hardware(HardwareError::Other));
                        };
                        // SAFETY: `fd` was just received via SCM_RIGHTS, so this
                        // process owns it exclusively; `OwnedDmaBuf` closes it once
                        // on drop.
                        let mut dmabuf =
                            unsafe { OwnedDmaBuf::from_raw(fd, rec.stride, rec.offset) };
                        if let Some(layout) = rec.plane_layout {
                            dmabuf = dmabuf.with_plane_layout(layout);
                        }
                        // Re-attach the shared stream semaphore + this frame's value
                        // so the GPU consumer waits before reading.
                        if rec.sync_value != 0 {
                            if let Some(s) = &sync {
                                dmabuf = dmabuf.with_sync(s.clone(), rec.sync_value);
                            }
                        }
                        let dmabuf = dmabuf.with_release_waker(Arc::clone(&release_waker));
                        lent.push((frames_received, dmabuf.clone()));
                        frames_received += 1;
                        out.push(PipelinePacket::DataFrame(Frame {
                            domain: MemoryDomain::DmaBuf(dmabuf),
                            timing: timeline.apply(rec.timing),
                            sequence: rec.sequence,
                            meta: Default::default(),
                        }))
                        .await?;
                        emitted += 1;
                        if emitted >= limit {
                            out.push(PipelinePacket::Eos).await?;
                            break;
                        }
                    }
                    _ => {
                        if let Some(fd) = fd {
                            close_fd(fd);
                        }
                    }
                }
            }
            Ok(emitted)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_round_trips() {
        let rec = Record {
            tag: TAG_FRAME,
            format: raw_format_to_u8(RawVideoFormat::Nv12),
            width: 1920,
            height: 1080,
            stride: 2048,
            offset: 4096,
            sequence: 99,
            timing: FrameTiming {
                pts_ns: 1_000_000,
                dts_ns: 900_000,
                duration_ns: 33_000,
                capture_ns: 5,
                arrival_ns: 6,
                keyframe: true,
                ..FrameTiming::default()
            },
            sync_value: 7,
            plane_layout: None,
        };
        let bytes = rec.encode();
        assert_eq!(bytes.len(), RECORD_LEN);
        assert_eq!(Record::decode(&bytes), Some(rec));
    }

    #[test]
    fn a_plane_count_past_the_record_fails_to_decode() {
        let mut bytes = Record::zeroed(TAG_FRAME).encode();
        bytes[PLANE_COUNT_AT] = u8::MAX;
        assert_eq!(Record::decode(&bytes), None);
    }

    fn null_dmabuf(stride: u32) -> OwnedDmaBuf {
        let fd = std::os::fd::IntoRawFd::into_raw_fd(std::fs::File::open("/dev/null").unwrap());
        // SAFETY: the descriptor was just opened and is handed over whole.
        unsafe { OwnedDmaBuf::from_raw(fd, stride, 0) }
    }

    fn dmabuf_frame(dmabuf: OwnedDmaBuf) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::DmaBuf(dmabuf),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        })
    }

    // a sink connected to a configured source, `first` sent along with the caps
    async fn connected_pair(
        name: &str,
        caps: &Caps,
        first: OwnedDmaBuf,
    ) -> (DmaBufSink, DmaBufSrc) {
        let path = std::env::temp_dir().join(alloc::format!(
            "g2g-localdmabuf-{name}-{}.sock",
            std::process::id()
        ));
        let path = path.to_str().unwrap();
        let mut source = DmaBufSrc::new(path);
        let _ = std::fs::remove_file(path);
        source.listener = Some(StdUnixListener::bind(path).unwrap());
        let mut sink = DmaBufSink::new(path);
        sink.configure_pipeline(caps).unwrap();
        let mut unused = crate::testutil::CollectSink::default();
        let (discovered, sent) = tokio::join!(
            source.intercept_caps(),
            sink.process(dmabuf_frame(first), &mut unused)
        );
        discovered.unwrap();
        sent.unwrap();
        source.configure_pipeline(caps).unwrap();
        let _ = std::fs::remove_file(path);
        (sink, source)
    }

    #[tokio::test]
    async fn a_dmabuf_arrives_with_the_plane_layout_it_was_sent_with() {
        let (width, height) = (16, 8);
        let decoded_rows = 16;
        let stride = 32;
        // chroma follows the decoded rows, past where plane 0's stride and the visible height put it
        let layout = PlaneLayout::new(&[
            Plane { offset: 0, stride },
            Plane {
                offset: stride * decoded_rows,
                stride,
            },
        ])
        .unwrap();
        let caps = caps_of(RawVideoFormat::Nv12, width, height);
        let dmabuf = null_dmabuf(stride as u32).with_plane_layout(layout);
        let (_sink, source) = connected_pair("layout", &caps, dmabuf).await;
        let mut received = crate::testutil::CollectSink::default();
        source.with_frame_limit(1).run(&mut received).await.unwrap();

        let Some(PipelinePacket::DataFrame(frame)) = received
            .packets
            .into_iter()
            .find(|packet| matches!(packet, PipelinePacket::DataFrame(_)))
        else {
            panic!("no frame arrived");
        };
        let MemoryDomain::DmaBuf(dmabuf) = frame.domain else {
            panic!("the frame is not a dma-buf");
        };
        assert_eq!(dmabuf.plane_layout(), Some(&layout));
    }

    struct DroppingSink;

    impl OutputSink for DroppingSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
            drop(packet_slot.take());
            core::task::Poll::Ready(Ok(g2g_core::PushOutcome::Accepted))
        }
    }

    #[tokio::test]
    async fn the_source_ends_at_eos_from_a_sink_that_already_exited() {
        let caps = caps_of(RawVideoFormat::Nv12, 16, 8);
        let path = std::env::temp_dir().join(alloc::format!(
            "g2g-localdmabuf-exited-{}.sock",
            std::process::id()
        ));
        let path = path.to_str().unwrap();
        let mut source = DmaBufSrc::new(path);
        let _ = std::fs::remove_file(path);
        source.listener = Some(StdUnixListener::bind(path).unwrap());
        let peer = UnixStream::connect(path).await.unwrap();
        let mut announce = Record::zeroed(TAG_CAPS);
        announce.format = raw_format_to_u8(RawVideoFormat::Nv12);
        (announce.width, announce.height) = (16, 8);
        send_record(&peer, &announce, None).await.unwrap();
        let frame = null_dmabuf(16);
        let mut record = Record::zeroed(TAG_FRAME);
        record.stride = 16;
        send_record(&peer, &record, Some(frame.as_raw()))
            .await
            .unwrap();
        send_record(&peer, &Record::zeroed(TAG_EOS), None)
            .await
            .unwrap();
        drop(peer);
        drop(frame);

        source.intercept_caps().await.unwrap();
        source.configure_pipeline(&caps).unwrap();
        let _ = std::fs::remove_file(path);
        let emitted = source.run(&mut DroppingSink).await;
        assert_eq!(emitted.ok(), Some(1));
    }

    struct ChannelSink(tokio::sync::mpsc::UnboundedSender<PipelinePacket>);

    impl OutputSink for ChannelSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            let _ = self.0.send(packet);
            core::task::Poll::Ready(Ok(g2g_core::PushOutcome::Accepted))
        }
    }

    #[tokio::test]
    async fn the_sink_keeps_a_frame_until_downstream_of_the_source_drops_it() {
        const DEADLINE: std::time::Duration = std::time::Duration::from_secs(20);
        let caps = caps_of(RawVideoFormat::Nv12, 16, 8);
        let dmabuf = null_dmabuf(16);
        let watch = dmabuf.clone();
        let (mut sink, mut source) = connected_pair("release", &caps, dmabuf).await;
        let (sender, mut packets) = tokio::sync::mpsc::unbounded_channel();
        let mut downstream = ChannelSink(sender);
        let script = async {
            let frame = loop {
                match packets.recv().await.expect("the source runs") {
                    PipelinePacket::DataFrame(frame) => break frame,
                    _ => continue,
                }
            };
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            assert_eq!(
                watch.share_count(),
                2,
                "the sink keeps the frame downstream holds"
            );
            drop(frame);
            let started = std::time::Instant::now();
            while watch.share_count() > 1 {
                assert!(
                    started.elapsed() < DEADLINE,
                    "the release never reached the sink"
                );
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
            let mut unused = crate::testutil::CollectSink::default();
            sink.process(PipelinePacket::Eos, &mut unused)
                .await
                .unwrap();
        };
        let (ran, ()) = tokio::join!(source.run(&mut downstream), script);
        ran.unwrap();
    }
}
