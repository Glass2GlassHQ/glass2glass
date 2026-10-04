//! GStreamer's fd-passing IPC pair (M1216, `unixfd` feature): [`UnixFdSink`] and
//! [`UnixFdSrc`] speak the protocol of `gst/unixfd` in gst-plugins-bad, so
//! either end can be a `gst-launch-1.0` process. The wire itself is in
//! [`crate::unixfdwire`].
//!
//! The sink listens on a unix socket and serves any number of clients. A
//! dma-buf frame goes out as its own fd, a system frame is copied into a fresh
//! memfd. Each client holds every buffer it was sent until it answers with
//! RELEASE_BUFFER or disconnects.
//!
//! The source connects, takes its caps from the first CAPS command, and maps
//! each memfd read-only into a system frame without copying. The release goes
//! out when that frame is dropped. Under `memory:DMABuf` caps it can hand the
//! dma-buf downstream instead, released once no frame shares it any more.
//!
//! Timestamps on the wire are absolute `CLOCK_MONOTONIC`, as in gst: the sink
//! sends running time plus its base time and path latency, moved from the
//! pipeline clock onto `CLOCK_MONOTONIC`, and the source maps them back and
//! subtracts its own base time. Without an elected clock either end stands in
//! the monotonic clock with the base time taken when it starts.
//!
//! Three deviations from gst. The sink sends its caps once the first frame says
//! whether they carry `memory:DMABuf`, rather than on accept. A system frame
//! gets a fresh memfd instead of one from a pool. The source retries its
//! connect for [`CONNECT_RETRY_WINDOW`], so either side can start first.

use core::ffi::{c_void, CStr};
use core::future::Future;
use core::pin::Pin;
use core::task::Poll;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::io::{ErrorKind, Read};
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use std::os::unix::fs::FileExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Mutex, PoisonError};

use tokio::io::unix::AsyncFd;
use tokio::sync::Notify;

use g2g_core::frame::Frame;
use g2g_core::log::io_err;
use g2g_core::memory::{DomainSet, MemoryDomainKind, OwnedDmaBuf, SystemSlice};
use g2g_core::meta::{Plane, PlaneLayout};
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ClockSync, ConfigureOutcome,
    Dim, ElementMetadata, FrameTiming, G2gError, LatencyReport, MemoryDomain, MonotonicClock,
    OutputSink, PadTemplate, PadTemplates, PipelineClock, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, RawVideoFormat,
};

use crate::capsfilter::{dmabuf_gst_caps, normalize_gst_caps, parse_caps, read_memory_feature};
use crate::dmabufmap::DmaBufReadMap;
use crate::paddedrows::{padded_planes, PaddedPlane};
use crate::pixel::pack_planes;
use crate::unixfdwire::{
    caps_payload, decode_caps, decode_release, from_wire_time, gst_video_format, monotonic_offset,
    protocol_error, read_command, release_payload, send_command, socket, take_command,
    to_wire_time, Memory, NewBuffer, Socket, SocketType, VideoMeta, VideoPlane,
    BUFFER_FLAG_DELTA_UNIT, CLOCK_TIME_NONE, COMMAND_CAPS, COMMAND_EOS, COMMAND_NEW_BUFFER,
    COMMAND_RELEASE_BUFFER, MEMORY_TYPE_DEFAULT, MEMORY_TYPE_DMABUF, SOCKET_TYPE_NAMES,
};

const DEFAULT_MIN_MEMORY_SIZE: i64 = 0;
const DEFAULT_MIN_MEMORY_SIZE_TEXT: &str = "0";
/// The `min-memory-size` that refuses to copy a system frame.
const COPY_DISABLED_TEXT: &str = "-1";
const MAX_INT64_TEXT: &str = "9223372036854775807";
const MEMFD_NAME: &CStr = c"g2g-unixfdsink";
/// Bytes read from a client socket per syscall. Only RELEASE_BUFFER comes back.
const CLIENT_READ_CHUNK: usize = 256;
const FIRST_BUFFER_ID: u64 = 1;
/// How long the source keeps retrying a connect the sink has not accepted yet.
pub const CONNECT_RETRY_WINDOW: Duration = Duration::from_secs(5);
const CONNECT_RETRY_GAP: Duration = Duration::from_millis(20);
/// How often the source looks for dma-buf frames downstream has dropped.
const DMABUF_RELEASE_POLL: Duration = Duration::from_millis(5);

fn raw_video_geometry(caps: &Caps) -> Option<(RawVideoFormat, usize, usize)> {
    match caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            ..
        } => Some((*format, *width as usize, *height as usize)),
        _ => None,
    }
}

/// The elected clock, or the monotonic clock based at this call when there is
/// none.
fn clock_or_monotonic(clock: &mut Option<ClockSync>) -> &ClockSync {
    clock.get_or_insert_with(|| ClockSync::new(Arc::new(MonotonicClock), MonotonicClock.now_ns()))
}

fn descriptor_size(fd: RawFd) -> Result<u64, G2gError> {
    let mut stat = core::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: fstat writes one `stat` into the live buffer and reads nothing else.
    if unsafe { libc::fstat(fd, stat.as_mut_ptr()) } != 0 {
        return Err(io_err(std::io::Error::last_os_error()));
    }
    // SAFETY: fstat returned 0, so it filled the struct.
    let stat = unsafe { stat.assume_init() };
    u64::try_from(stat.st_size).map_err(|_| protocol_error("descriptor reports a negative size"))
}

fn memfd_holding(bytes: &[u8], size: usize) -> Result<OwnedFd, G2gError> {
    // SAFETY: the name is a NUL-terminated literal and the flag is a documented
    // memfd_create bit.
    let raw = unsafe { libc::memfd_create(MEMFD_NAME.as_ptr(), libc::MFD_CLOEXEC) };
    if raw < 0 {
        return Err(io_err(std::io::Error::last_os_error()));
    }
    // SAFETY: memfd_create returned a fresh descriptor nothing else owns.
    let file = std::fs::File::from(unsafe { OwnedFd::from_raw_fd(raw) });
    file.set_len(size as u64).map_err(io_err)?;
    file.write_all_at(bytes, 0).map_err(io_err)?;
    Ok(file.into())
}

fn video_meta(
    format: RawVideoFormat,
    width: usize,
    height: usize,
    planes: impl Iterator<Item = (usize, usize)>,
) -> Option<VideoMeta> {
    let planes = planes
        .map(|(offset, stride)| {
            Some(VideoPlane {
                offset: u64::try_from(offset).ok()?,
                stride: i32::try_from(stride).ok()?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(VideoMeta {
        flags: 0,
        format: gst_video_format(format)?,
        width: u32::try_from(width).ok()?,
        height: u32::try_from(height).ok()?,
        planes,
    })
}

fn padded_offsets_and_strides(planes: Vec<PaddedPlane>) -> impl Iterator<Item = (usize, usize)> {
    planes.into_iter().map(|plane| (plane.offset, plane.stride))
}

/// What keeps a sent buffer's memory alive until every client releases it.
#[derive(Debug)]
enum SentMemory {
    DmaBuf { _frame: Frame },
    Memfd { _memfd: OwnedFd },
}

#[derive(Debug)]
struct Client {
    stream: Socket,
    received: Vec<u8>,
    sent_caps: Option<String>,
    held: Vec<(u64, Arc<SentMemory>)>,
}

impl Client {
    fn new(stream: Socket) -> Self {
        Self {
            stream,
            received: Vec::new(),
            sent_caps: None,
            held: Vec::new(),
        }
    }

    async fn send_caps_if_stale(&mut self, caps: Option<&str>) -> std::io::Result<()> {
        let Some(caps) = caps else {
            return Ok(());
        };
        if self.sent_caps.as_deref() == Some(caps) {
            return Ok(());
        }
        send_command(&self.stream, COMMAND_CAPS, &caps_payload(caps), &[]).await?;
        self.sent_caps = Some(caps.to_string());
        Ok(())
    }

    /// Apply every release waiting on the socket. `false` means this client is
    /// finished with: it closed, errored, named a buffer it does not hold, or
    /// sent a command only a sink sends.
    fn read_releases(&mut self) -> bool {
        loop {
            let mut chunk = [0u8; CLIENT_READ_CHUNK];
            let filled = match self.stream.get_ref().read(&mut chunk) {
                Ok(0) => return false,
                Ok(filled) => filled,
                Err(error) if error.kind() == ErrorKind::WouldBlock => return true,
                Err(_) => return false,
            };
            self.received.extend_from_slice(&chunk[..filled]);
            loop {
                match take_command(&mut self.received) {
                    Err(_) => return false,
                    Ok(None) => break,
                    Ok(Some((COMMAND_RELEASE_BUFFER, payload))) => {
                        let Some(id) = decode_release(&payload) else {
                            return false;
                        };
                        let Some(index) = self.held.iter().position(|(held, _)| *held == id) else {
                            return false;
                        };
                        self.held.swap_remove(index);
                    }
                    Ok(Some((COMMAND_NEW_BUFFER | COMMAND_CAPS, _))) => return false,
                    // gst ignores a command it does not know, the protocol may grow
                    Ok(Some(_)) => {}
                }
            }
        }
    }
}

/// Send one command to every client, after the caps if a client has not seen
/// them yet. A client the send fails on is dropped, releasing what it held.
async fn send_to_clients(
    clients: &mut Vec<Client>,
    caps: Option<&str>,
    command: u32,
    payload: &[u8],
    fds: &[RawFd],
    held: Option<(u64, &Arc<SentMemory>)>,
) {
    let mut index = 0;
    while index < clients.len() {
        let client = &mut clients[index];
        let delivered = match client.send_caps_if_stale(caps).await {
            Ok(()) => send_command(&client.stream, command, payload, fds)
                .await
                .is_ok(),
            Err(_) => false,
        };
        if !delivered {
            clients.swap_remove(index);
            continue;
        }
        if let Some((id, memory)) = held {
            client.held.push((id, Arc::clone(memory)));
        }
        index += 1;
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::unixfd::UnixFdSink;
///
/// // gst-launch equivalent: unixfdsink socket-path=/tmp/g2g-unixfd
/// let sink = UnixFdSink::new("/tmp/g2g-unixfd");
/// ```
#[derive(Debug)]
pub struct UnixFdSink {
    socket_path: String,
    socket_type: SocketType,
    wait_for_connection: bool,
    min_memory_size: i64,
    bound: Option<UnixListener>,
    listener: Option<AsyncFd<UnixListener>>,
    clients: Vec<Client>,
    caps: Option<Caps>,
    /// Whether the input is dma-buf, fixed by the first frame. The caps a client
    /// is sent depend on it, so they wait for it.
    dmabuf_input: Option<bool>,
    clock: Option<ClockSync>,
    next_buffer_id: u64,
}

impl Default for UnixFdSink {
    fn default() -> Self {
        Self {
            socket_path: String::new(),
            socket_type: SocketType::Path,
            wait_for_connection: false,
            min_memory_size: DEFAULT_MIN_MEMORY_SIZE,
            bound: None,
            listener: None,
            clients: Vec::new(),
            caps: None,
            dmabuf_input: None,
            clock: None,
            next_buffer_id: FIRST_BUFFER_ID,
        }
    }
}

impl UnixFdSink {
    pub fn new(socket_path: impl Into<String>) -> Self {
        let mut sink = Self::default();
        sink.socket_path = socket_path.into();
        sink
    }

    pub fn with_socket_type(mut self, socket_type: SocketType) -> Self {
        self.socket_type = socket_type;
        self
    }

    pub fn with_wait_for_connection(mut self, wait: bool) -> Self {
        self.wait_for_connection = wait;
        self
    }

    /// Bind the socket now, so a peer can connect before the pipeline runs.
    /// `configure_pipeline` calls it when nothing is bound yet.
    pub fn open(&mut self) -> Result<(), G2gError> {
        if self.is_open() {
            return Ok(());
        }
        if self.socket_path.is_empty() {
            return Err(G2gError::NotConfigured);
        }
        let bound = self.socket_type.bind(&self.socket_path).map_err(io_err)?;
        bound.set_nonblocking(true).map_err(io_err)?;
        self.bound = Some(bound);
        Ok(())
    }

    fn is_open(&self) -> bool {
        self.bound.is_some() || self.listener.is_some()
    }

    /// Register the bound socket with the runtime, which only works inside it.
    fn adopt(&mut self) -> Result<&AsyncFd<UnixListener>, G2gError> {
        if let Some(bound) = self.bound.take() {
            self.listener = Some(AsyncFd::new(bound).map_err(io_err)?);
        }
        self.listener.as_ref().ok_or(G2gError::NotConfigured)
    }

    fn caps_text(&self) -> Result<Option<String>, G2gError> {
        let (Some(caps), Some(dmabuf)) = (&self.caps, self.dmabuf_input) else {
            return Ok(None);
        };
        match dmabuf {
            true => dmabuf_gst_caps(caps)
                .map(Some)
                .ok_or(G2gError::CapsMismatch),
            false => Ok(Some(caps.to_gst_string())),
        }
    }

    async fn attach(&mut self, stream: UnixStream) -> Result<(), G2gError> {
        let mut client = Client::new(socket(stream).map_err(io_err)?);
        if client
            .send_caps_if_stale(self.caps_text()?.as_deref())
            .await
            .is_ok()
        {
            self.clients.push(client);
        }
        Ok(())
    }

    /// Take every client already waiting in the listen backlog, straight off
    /// the socket so one the runtime has not noticed yet is not missed.
    async fn accept_pending(&mut self) -> Result<(), G2gError> {
        loop {
            let stream = match self.adopt()?.get_ref().accept() {
                Ok((stream, _peer)) => stream,
                Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(()),
                Err(error) => return Err(io_err(error)),
            };
            self.attach(stream).await?;
        }
    }

    async fn wait_for_client(&mut self) -> Result<(), G2gError> {
        loop {
            let mut ready = self.adopt()?.readable().await.map_err(io_err)?;
            let accepted = ready.try_io(|listener| listener.get_ref().accept());
            drop(ready);
            if let Ok(accepted) = accepted {
                let (stream, _peer) = accepted.map_err(io_err)?;
                return self.attach(stream).await;
            }
        }
    }

    fn drain_releases(&mut self) {
        self.clients.retain_mut(Client::read_releases);
    }

    /// Accept new clients and read what the connected ones sent back.
    async fn service_clients(&mut self) -> Result<(), G2gError> {
        self.accept_pending().await?;
        self.drain_releases();
        Ok(())
    }

    async fn send_frame(&mut self, frame: Frame) -> Result<(), G2gError> {
        self.service_clients().await?;
        let dmabuf = matches!(frame.domain, MemoryDomain::DmaBuf(_));
        // the wire cannot carry the fence, so a reader could see an unfinished GPU write
        if matches!(&frame.domain, MemoryDomain::DmaBuf(fenced) if fenced.sync_fd().is_some()) {
            return Err(G2gError::UnsupportedDomain);
        }
        if self.dmabuf_input.is_some_and(|seen| seen != dmabuf) {
            return Err(G2gError::UnsupportedDomain);
        }
        self.dmabuf_input = Some(dmabuf);
        if !dmabuf && self.min_memory_size < 0 {
            return Err(G2gError::UnsupportedDomain);
        }
        while self.wait_for_connection && self.clients.is_empty() {
            self.wait_for_client().await?;
        }
        if self.clients.is_empty() {
            return Ok(());
        }
        let (buffer, memory, fd) = self.encode_frame(frame)?;
        let caps = self.caps_text()?;
        send_to_clients(
            &mut self.clients,
            caps.as_deref(),
            COMMAND_NEW_BUFFER,
            &buffer.encode(),
            &[fd],
            Some((buffer.id, &memory)),
        )
        .await;
        Ok(())
    }

    fn encode_frame(
        &mut self,
        frame: Frame,
    ) -> Result<(NewBuffer, Arc<SentMemory>, RawFd), G2gError> {
        let caps = self.caps.as_ref().ok_or(G2gError::NotConfigured)?;
        let clock = clock_or_monotonic(&mut self.clock);
        let (base_time, latency) = (clock.base_time(), clock.path_latency_min_ns());
        let offset = monotonic_offset(clock.now_ns());
        let wire_time = |running| to_wire_time(running, base_time, latency, offset);
        let delta_unit = !frame.timing.keyframe && matches!(caps, Caps::CompressedVideo { .. });
        let mut buffer = NewBuffer {
            id: self.next_buffer_id,
            pts: wire_time(frame.timing.pts_ns),
            dts: wire_time(frame.timing.dts_ns),
            duration: match frame.timing.duration_ns {
                0 => CLOCK_TIME_NONE,
                duration => duration,
            },
            offset: CLOCK_TIME_NONE,
            offset_end: CLOCK_TIME_NONE,
            flags: if delta_unit {
                BUFFER_FLAG_DELTA_UNIT
            } else {
                0
            },
            memory_type: MEMORY_TYPE_DEFAULT,
            memories: Vec::new(),
            video_meta: None,
        };
        self.next_buffer_id += 1;
        let geometry = raw_video_geometry(caps);
        let (memory, fd) = match &frame.domain {
            MemoryDomain::DmaBuf(dmabuf) => {
                let fd = dmabuf.as_raw();
                let offset = u64::from(dmabuf.offset);
                let size = descriptor_size(fd)?
                    .checked_sub(offset)
                    .ok_or(G2gError::UnsupportedDomain)?;
                buffer.memory_type = MEMORY_TYPE_DMABUF;
                buffer.memories.push(Memory { size, offset });
                buffer.video_meta = geometry.and_then(|(format, width, height)| {
                    let planes = padded_planes(format, width, height, 0, dmabuf.stride as usize)?;
                    video_meta(format, width, height, padded_offsets_and_strides(planes))
                });
                (Arc::new(SentMemory::DmaBuf { _frame: frame }), fd)
            }
            domain => {
                let bytes = domain.require_system_bytes(g2g_core::log::short_type_name::<Self>())?;
                let minimum = usize::try_from(self.min_memory_size).unwrap_or(usize::MAX);
                let allocated = bytes.len().max(minimum);
                let memfd = memfd_holding(&bytes, allocated)?;
                buffer.memories.push(Memory {
                    size: bytes.len() as u64,
                    offset: 0,
                });
                buffer.video_meta = geometry.and_then(|(format, width, height)| {
                    let planes: Vec<(usize, usize)> = match frame.meta.get::<PlaneLayout>() {
                        Some(layout) => (0..layout.count())
                            .filter_map(|index| layout.plane(index))
                            .map(|plane| (plane.offset, plane.stride))
                            .collect(),
                        None => {
                            padded_offsets_and_strides(padded_planes(format, width, height, 0, 0)?)
                                .collect()
                        }
                    };
                    video_meta(format, width, height, planes.into_iter())
                });
                let fd = memfd.as_raw_fd();
                (Arc::new(SentMemory::Memfd { _memfd: memfd }), fd)
            }
        };
        Ok((buffer, memory, fd))
    }

    async fn refresh_caps(&mut self) -> Result<(), G2gError> {
        self.service_clients().await?;
        let caps = self.caps_text()?;
        let mut index = 0;
        while index < self.clients.len() {
            if self.clients[index]
                .send_caps_if_stale(caps.as_deref())
                .await
                .is_err()
            {
                self.clients.swap_remove(index);
                continue;
            }
            index += 1;
        }
        Ok(())
    }

    async fn send_eos(&mut self) -> Result<(), G2gError> {
        self.service_clients().await?;
        let caps = self.caps_text()?;
        send_to_clients(
            &mut self.clients,
            caps.as_deref(),
            COMMAND_EOS,
            &[],
            &[],
            None,
        )
        .await;
        Ok(())
    }
}

impl Drop for UnixFdSink {
    fn drop(&mut self) {
        if self.socket_type == SocketType::Path && self.is_open() {
            let _ = std::fs::remove_file(&self.socket_path);
        }
    }
}

impl AsyncElement for UnixFdSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// A dma-buf goes out as its own fd. A system frame is copied into a memfd,
    /// unless `min-memory-size` is -1.
    fn input_domains(&self) -> DomainSet {
        let dmabuf = DomainSet::only(MemoryDomainKind::DmaBuf);
        match self.min_memory_size < 0 {
            true => dmabuf,
            false => dmabuf.with(MemoryDomainKind::System),
        }
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.open()?;
        self.caps = Some(absolute_caps.clone());
        clock_or_monotonic(&mut self.clock);
        Ok(ConfigureOutcome::Accepted)
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.clock = Some(sync);
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => self.send_frame(frame).await,
                PipelinePacket::CapsChanged(caps) => {
                    self.caps = Some(caps);
                    self.refresh_caps().await
                }
                PipelinePacket::Eos => self.send_eos().await,
                _ => Ok(()),
            }
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SINK_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Unix file descriptor sink",
            "Sink/IPC",
            "Sends frames as memfd or dma-buf descriptors over a unix socket to a GStreamer-compatible unixfdsrc",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "socket-path" | "socket-type" if self.is_open() => Err(PropError::Value),
            "socket-path" => {
                self.socket_path = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "socket-type" => {
                let name = value.as_str().ok_or(PropError::Type)?;
                self.socket_type = SocketType::from_name(name).ok_or(PropError::Value)?;
                Ok(())
            }
            "wait-for-connection" => {
                self.wait_for_connection = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "min-memory-size" => {
                let size = value.as_int().ok_or(PropError::Type)?;
                if size < -1 {
                    return Err(PropError::Value);
                }
                self.min_memory_size = size;
                Ok(())
            }
            "num-clients" => Err(PropError::ReadOnly),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "socket-path" => Some(PropValue::Str(self.socket_path.clone())),
            "socket-type" => Some(PropValue::Str(self.socket_type.name().to_string())),
            "wait-for-connection" => Some(PropValue::Bool(self.wait_for_connection)),
            "min-memory-size" => Some(PropValue::Int(self.min_memory_size)),
            "num-clients" => Some(PropValue::Uint(self.clients.len() as u64)),
            _ => None,
        }
    }
}

static SINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "socket-path",
        PropKind::Str,
        "path or abstract name of the unix socket the clients connect to",
    ),
    PropertySpec::new("socket-type", PropKind::Str, "where the socket lives")
        .with_enum_values(SOCKET_TYPE_NAMES)
        .with_default(SocketType::Path.name()),
    PropertySpec::new(
        "wait-for-connection",
        PropKind::Bool,
        "hold the stream until a client connects, instead of dropping frames",
    )
    .with_default("false"),
    PropertySpec::new(
        "min-memory-size",
        PropKind::Int,
        "smallest memfd a system frame is copied into (-1 refuses to copy)",
    )
    .with_default(DEFAULT_MIN_MEMORY_SIZE_TEXT)
    .with_range(COPY_DISABLED_TEXT, MAX_INT64_TEXT),
    PropertySpec::new("num-clients", PropKind::Uint, "clients connected now").read_only(),
];

impl PadTemplates for UnixFdSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}

/// Releases the source owes the sink, queued from wherever a frame drops.
#[derive(Debug, Default)]
struct Releases {
    ids: Mutex<Vec<u64>>,
    wake: Notify,
}

impl Releases {
    fn push(&self, id: u64) {
        self.ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(id);
        self.wake.notify_one();
    }

    fn take(&self) -> Vec<u64> {
        core::mem::take(&mut *self.ids.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

/// A received memory mapped into a system frame. Dropped with the frame.
#[derive(Debug)]
struct MappedMemory {
    map: DmaBufReadMap,
    descriptor: OwnedDmaBuf,
    id: u64,
    releases: Arc<Releases>,
}

unsafe extern "C" fn release_mapped_memory(user: *mut c_void) {
    // SAFETY: `user` is the box `frame_memory` leaked for this one slice, and the
    // slice calls this once, on drop.
    let mapped = unsafe { Box::from_raw(user.cast::<MappedMemory>()) };
    let MappedMemory {
        map,
        descriptor,
        id,
        releases,
    } = *mapped;
    drop(map);
    drop(descriptor);
    releases.push(id);
}

/// Where the rows a `GstVideoMeta` describes lie, `None` when they are packed
/// tight. Checked against `len` bytes, so every row it names lies inside them.
fn padded_layout(
    caps: &Caps,
    meta: Option<&VideoMeta>,
    len: usize,
) -> Result<Option<PlaneLayout>, G2gError> {
    let (Some(meta), Some((format, width, height))) = (meta, raw_video_geometry(caps)) else {
        return Ok(None);
    };
    let tight = padded_planes(format, width, height, 0, 0).ok_or(G2gError::CapsMismatch)?;
    let planes = meta
        .planes
        .iter()
        .map(|plane| {
            Some(Plane {
                offset: usize::try_from(plane.offset).ok()?,
                stride: usize::try_from(plane.stride).ok()?,
            })
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| protocol_error("video meta plane is out of range"))?;
    if planes.len() != tight.len() {
        return Err(protocol_error(
            "video meta plane count does not match the caps",
        ));
    }
    let is_tight = planes.iter().zip(&tight).all(|(plane, expected)| {
        plane.offset == expected.offset && plane.stride == expected.stride
    });
    if is_tight {
        return Ok(None);
    }
    let layout =
        PlaneLayout::new(&planes).ok_or_else(|| protocol_error("video meta has no planes"))?;
    for (index, plane) in tight.iter().enumerate() {
        let last_row = plane.rows.saturating_sub(1);
        let fits = layout
            .row_range(index, last_row, plane.row_bytes)
            .is_some_and(|rows| rows.end <= len);
        if !fits {
            return Err(protocol_error("video meta rows lie outside the memory"));
        }
    }
    Ok(Some(layout))
}

/// Received rows to hand on: as they lie, with the padded layout to declare
/// when there is one, or packed tight for a downstream that did not ask for it.
#[derive(Debug)]
enum Rows {
    AsTheyLie(Option<PlaneLayout>),
    Packed(Box<[u8]>),
}

/// The plane-0 stride and offset a dma-buf frame carries, read from the
/// `GstVideoMeta` when there is one. g2g's dma-buf layout puts the planes back
/// to back after plane 0, so a meta describing anything else is refused.
fn dmabuf_layout(caps: &Caps, meta: Option<&VideoMeta>) -> Result<(u32, u64), G2gError> {
    let Some((format, width, height)) = raw_video_geometry(caps) else {
        return Ok((0, 0));
    };
    let Some(meta) = meta else {
        let stride = format
            .plane_stride(0, width as u32)
            .ok_or(G2gError::CapsMismatch)?;
        return Ok((stride, 0));
    };
    let first = meta
        .planes
        .first()
        .ok_or_else(|| protocol_error("video meta has no planes"))?;
    let stride = u32::try_from(first.stride).map_err(|_| protocol_error("negative stride"))?;
    let first_offset =
        usize::try_from(first.offset).map_err(|_| protocol_error("plane offset out of range"))?;
    let expected = padded_planes(format, width, height, first_offset, stride as usize)
        .ok_or(G2gError::CapsMismatch)?;
    let matches = expected.len() == meta.planes.len()
        && expected.iter().zip(&meta.planes).all(|(want, got)| {
            u64::try_from(want.offset).is_ok_and(|offset| offset == got.offset)
                && i32::try_from(want.stride).is_ok_and(|stride| stride == got.stride)
        });
    if !matches {
        return Err(G2gError::CapsMismatch);
    }
    Ok((stride, first.offset))
}

fn require_covered(fd: &OwnedFd, memory: &Memory) -> Result<(), G2gError> {
    let end = memory
        .offset
        .checked_add(memory.size)
        .ok_or_else(|| protocol_error("memory range overflows"))?;
    if end > descriptor_size(fd.as_raw_fd())? {
        return Err(protocol_error("memory lies past the end of its descriptor"));
    }
    Ok(())
}

/// Map `memory` of `fd` read-only, checking first that the descriptor holds it.
fn map_memory(fd: OwnedFd, memory: &Memory) -> Result<(DmaBufReadMap, OwnedDmaBuf), G2gError> {
    require_covered(&fd, memory)?;
    let offset =
        u32::try_from(memory.offset).map_err(|_| protocol_error("memory offset out of range"))?;
    // SAFETY: `fd` came off the socket and this process owns it alone.
    let descriptor = unsafe { OwnedDmaBuf::from_raw(fd.into_raw_fd(), 0, offset) };
    let map = DmaBufReadMap::read(&descriptor)?;
    Ok((map, descriptor))
}

fn mapped_bytes<'a>(map: &'a DmaBufReadMap, memory: &Memory) -> Result<&'a [u8], G2gError> {
    let size =
        usize::try_from(memory.size).map_err(|_| protocol_error("memory size out of range"))?;
    map.as_slice()
        .get(..size)
        .ok_or_else(|| protocol_error("memory lies past the end of its mapping"))
}

fn owned_frame_bytes(bytes: Box<[u8]>) -> MemoryDomain {
    MemoryDomain::System(SystemSlice::from_boxed(bytes))
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::unixfd::UnixFdSrc;
///
/// // gst-launch equivalent: unixfdsrc socket-path=/tmp/g2g-unixfd
/// let source = UnixFdSrc::new("/tmp/g2g-unixfd");
/// ```
#[derive(Debug)]
pub struct UnixFdSrc {
    socket_path: String,
    socket_type: SocketType,
    stream: Option<Socket>,
    caps: Option<Caps>,
    dmabuf_caps: bool,
    /// The domain the allocation cascade settled on, when it ran.
    allocated_domain: Option<MemoryDomainKind>,
    keep_row_padding: bool,
    clock: Option<ClockSync>,
}

impl Default for UnixFdSrc {
    fn default() -> Self {
        Self {
            socket_path: String::new(),
            socket_type: SocketType::Path,
            stream: None,
            caps: None,
            dmabuf_caps: false,
            allocated_domain: None,
            keep_row_padding: false,
            clock: None,
        }
    }
}

impl UnixFdSrc {
    pub fn new(socket_path: impl Into<String>) -> Self {
        Self {
            socket_path: socket_path.into(),
            ..Self::default()
        }
    }

    pub fn with_socket_type(mut self, socket_type: SocketType) -> Self {
        self.socket_type = socket_type;
        self
    }

    async fn connect(&self) -> Result<Socket, G2gError> {
        let deadline = tokio::time::Instant::now() + CONNECT_RETRY_WINDOW;
        loop {
            match self.socket_type.connect(&self.socket_path) {
                Ok(stream) => return socket(stream).map_err(io_err),
                Err(error) if tokio::time::Instant::now() >= deadline => return Err(io_err(error)),
                Err(_) => tokio::time::sleep(CONNECT_RETRY_GAP).await,
            }
        }
    }

    async fn peer_caps(&mut self) -> Result<Caps, G2gError> {
        if let Some(caps) = &self.caps {
            return Ok(caps.clone());
        }
        let stream = self.connect().await?;
        loop {
            let command = read_command(&stream)
                .await?
                .ok_or_else(|| protocol_error("sink closed the socket before sending caps"))?;
            match command.command {
                COMMAND_CAPS => {
                    let (caps, dmabuf) = read_caps(&command.payload)?;
                    self.caps = Some(caps.clone());
                    self.dmabuf_caps = dmabuf;
                    self.stream = Some(stream);
                    return Ok(caps);
                }
                COMMAND_NEW_BUFFER | COMMAND_EOS | COMMAND_RELEASE_BUFFER => {
                    return Err(protocol_error("sink sent a command before its caps"))
                }
                _ => {}
            }
        }
    }

    fn emits_dmabuf(&self) -> bool {
        self.dmabuf_caps
            && self.allocated_domain.unwrap_or(self.output_memory()) == MemoryDomainKind::DmaBuf
    }

    fn frame_timing(&mut self, buffer: &NewBuffer) -> FrameTiming {
        let clock = clock_or_monotonic(&mut self.clock);
        let base_time = clock.base_time();
        let offset = monotonic_offset(clock.now_ns());
        let pts = from_wire_time(buffer.pts, base_time, offset);
        FrameTiming {
            pts_ns: pts,
            dts_ns: match buffer.dts {
                CLOCK_TIME_NONE => pts,
                dts => from_wire_time(dts, base_time, offset),
            },
            duration_ns: match buffer.duration {
                CLOCK_TIME_NONE => 0,
                duration => duration,
            },
            capture_ns: 0,
            arrival_ns: g2g_core::metrics::monotonic_ns(),
            keyframe: buffer.flags & BUFFER_FLAG_DELTA_UNIT == 0,
        }
    }

    fn settle_rows(&self, bytes: &[u8], meta: Option<&VideoMeta>) -> Result<Rows, G2gError> {
        let caps = self.caps.as_ref().ok_or(G2gError::NotConfigured)?;
        let Some(layout) = padded_layout(caps, meta, bytes.len())? else {
            return Ok(Rows::AsTheyLie(None));
        };
        if self.keep_row_padding {
            return Ok(Rows::AsTheyLie(Some(layout)));
        }
        let (format, width, height) = raw_video_geometry(caps).ok_or(G2gError::CapsMismatch)?;
        let packed =
            pack_planes(bytes, format, width, height, &layout).ok_or(G2gError::CapsMismatch)?;
        Ok(Rows::Packed(packed))
    }

    /// The frame memory for one NEW_BUFFER, and the padded layout to declare on
    /// it. A buffer whose memory is done with at once (copied, or empty) is
    /// queued for release here.
    fn frame_memory(
        &self,
        buffer: &NewBuffer,
        mut fds: Vec<OwnedFd>,
        releases: &Arc<Releases>,
        held_dmabufs: &mut Vec<(u64, OwnedDmaBuf)>,
    ) -> Result<(MemoryDomain, Option<PlaneLayout>), G2gError> {
        let missing_fd = || protocol_error("descriptor count differs from the memory count");
        if fds.len() != buffer.memories.len() {
            return Err(missing_fd());
        }
        let caps = self.caps.as_ref().ok_or(G2gError::NotConfigured)?;
        let meta = buffer.video_meta.as_ref();
        if buffer.memories.is_empty() {
            releases.push(buffer.id);
            return Ok((owned_frame_bytes(Box::default()), None));
        }
        if self.emits_dmabuf() {
            let [memory] = buffer.memories.as_slice() else {
                return Err(G2gError::UnsupportedDomain);
            };
            let fd = fds.pop().ok_or_else(missing_fd)?;
            require_covered(&fd, memory)?;
            let (stride, plane_offset) = dmabuf_layout(caps, meta)?;
            let offset = memory
                .offset
                .checked_add(plane_offset)
                .and_then(|offset| u32::try_from(offset).ok())
                .ok_or_else(|| protocol_error("plane offset out of range"))?;
            // SAFETY: `fd` came off the socket and this process owns it alone.
            let dmabuf = unsafe { OwnedDmaBuf::from_raw(fd.into_raw_fd(), stride, offset) };
            held_dmabufs.push((buffer.id, dmabuf.clone()));
            return Ok((MemoryDomain::DmaBuf(dmabuf), None));
        }
        if let [memory] = buffer.memories.as_slice() {
            let fd = fds.pop().ok_or_else(missing_fd)?;
            let (map, descriptor) = map_memory(fd, memory)?;
            let bytes = mapped_bytes(&map, memory)?;
            let layout = match self.settle_rows(bytes, meta)? {
                Rows::Packed(packed) => {
                    releases.push(buffer.id);
                    return Ok((owned_frame_bytes(packed), None));
                }
                Rows::AsTheyLie(layout) => layout,
            };
            let (pointer, len) = (bytes.as_ptr(), bytes.len());
            let mapped = Box::new(MappedMemory {
                map,
                descriptor,
                id: buffer.id,
                releases: Arc::clone(releases),
            });
            // SAFETY: the bytes are the mapping `mapped` owns, read-only and
            // unmapped only by `release_mapped_memory` when the slice drops.
            // munmap and the release queue are safe from any thread.
            let slice = unsafe {
                SystemSlice::from_foreign(
                    pointer,
                    len,
                    Some(release_mapped_memory),
                    Box::into_raw(mapped).cast(),
                )
            };
            return Ok((MemoryDomain::System(slice), layout));
        }
        let mut joined = Vec::new();
        for (memory, fd) in buffer.memories.iter().zip(fds) {
            let (map, _descriptor) = map_memory(fd, memory)?;
            joined.extend_from_slice(mapped_bytes(&map, memory)?);
        }
        releases.push(buffer.id);
        Ok(match self.settle_rows(&joined, meta)? {
            Rows::Packed(packed) => (owned_frame_bytes(packed), None),
            Rows::AsTheyLie(layout) => (owned_frame_bytes(joined.into_boxed_slice()), layout),
        })
    }

    fn receive_frame(
        &mut self,
        payload: &[u8],
        fds: Vec<OwnedFd>,
        sequence: u64,
        releases: &Arc<Releases>,
        held_dmabufs: &mut Vec<(u64, OwnedDmaBuf)>,
    ) -> Result<Frame, G2gError> {
        let buffer = NewBuffer::decode(payload)
            .ok_or_else(|| protocol_error("malformed NEW_BUFFER payload"))?;
        let (domain, layout) = self.frame_memory(&buffer, fds, releases, held_dmabufs)?;
        let mut frame = Frame::new(domain, self.frame_timing(&buffer), sequence);
        if let Some(layout) = layout {
            frame.meta.attach(layout);
        }
        Ok(frame)
    }
}

fn read_caps(payload: &[u8]) -> Result<(Caps, bool), G2gError> {
    let text =
        decode_caps(payload).ok_or_else(|| protocol_error("caps are not NUL-terminated text"))?;
    let memory = read_memory_feature(&normalize_gst_caps(text)).ok_or(G2gError::CapsMismatch)?;
    let caps = parse_caps(&memory.caps).ok_or(G2gError::CapsMismatch)?;
    Ok((caps, memory.dmabuf))
}

/// Send what the source owes: every release a dropped system frame queued, and
/// one for each dma-buf no frame shares any more.
async fn send_releases(
    stream: &Socket,
    releases: &Releases,
    held_dmabufs: &mut Vec<(u64, OwnedDmaBuf)>,
) {
    let mut ids = releases.take();
    held_dmabufs.retain(|(id, dmabuf)| {
        let shared = dmabuf.share_count() > 1;
        if !shared {
            ids.push(*id);
        }
        shared
    });
    for id in ids {
        // gst only warns when a release cannot go out, the next read reports a
        // closed socket
        let _ = send_command(stream, COMMAND_RELEASE_BUFFER, &release_payload(id), &[]).await;
    }
}

/// Wake when the sink sent something (`true`), a frame was dropped, or (while
/// dma-bufs are out) the poll period passed.
async fn wait_for_work(stream: &Socket, releases: &Releases, dmabufs_out: bool) -> bool {
    let mut dropped = core::pin::pin!(releases.wake.notified());
    let mut poll_period = core::pin::pin!(tokio::time::sleep(DMABUF_RELEASE_POLL));
    core::future::poll_fn(|cx| {
        if stream.poll_read_ready(cx).is_ready() {
            return Poll::Ready(true);
        }
        let other_work = dropped.as_mut().poll(cx).is_ready()
            || (dmabufs_out && poll_period.as_mut().poll(cx).is_ready());
        match other_work {
            true => Poll::Ready(false),
            false => Poll::Pending,
        }
    })
    .await
}

impl SourceLoop for UnixFdSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = Pin<Box<dyn Future<Output = Result<Caps, G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        Box::pin(self.peer_caps())
    }

    async fn caps_constraint(&mut self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(
            self.peer_caps().await?,
        )))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.caps.is_none() {
            return Err(G2gError::CapsMismatch);
        }
        Ok(ConfigureOutcome::Accepted)
    }

    fn latency(&self) -> LatencyReport {
        LatencyReport::live(0, None)
    }

    /// Plain caps are mapped memfds. `memory:DMABuf` caps hand the dma-buf on,
    /// or map it when downstream needs system memory. Before the caps arrive
    /// either is possible.
    fn output_domains(&self) -> DomainSet {
        let system = DomainSet::only(MemoryDomainKind::System);
        match (&self.caps, self.dmabuf_caps) {
            (Some(_), false) => system,
            _ => DomainSet::only(MemoryDomainKind::DmaBuf).with(MemoryDomainKind::System),
        }
    }

    fn output_memory(&self) -> MemoryDomainKind {
        self.output_domains()
            .preferred()
            .unwrap_or(MemoryDomainKind::System)
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        self.keep_row_padding = params.meta_requests.wants::<PlaneLayout>();
        let dmabuf = self.dmabuf_caps
            && (params.domain == MemoryDomainKind::DmaBuf
                || !params.accepts.contains(MemoryDomainKind::System));
        self.allocated_domain = Some(match dmabuf {
            true => MemoryDomainKind::DmaBuf,
            false => MemoryDomainKind::System,
        });
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.clock = Some(sync);
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let caps = self.caps.clone().ok_or(G2gError::NotConfigured)?;
            let stream = self.stream.take().ok_or(G2gError::NotConfigured)?;
            clock_or_monotonic(&mut self.clock);
            out.push(PipelinePacket::CapsChanged(caps)).await?;
            let releases = Arc::new(Releases::default());
            let mut held_dmabufs = Vec::new();
            let mut sequence = 0u64;
            loop {
                send_releases(&stream, &releases, &mut held_dmabufs).await;
                if !wait_for_work(&stream, &releases, !held_dmabufs.is_empty()).await {
                    continue;
                }
                let command = read_command(&stream)
                    .await?
                    .ok_or_else(|| protocol_error("sink closed the socket without EOS"))?;
                match command.command {
                    COMMAND_NEW_BUFFER => {
                        let frame = self.receive_frame(
                            &command.payload,
                            command.fds,
                            sequence,
                            &releases,
                            &mut held_dmabufs,
                        )?;
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                        sequence += 1;
                    }
                    COMMAND_CAPS => {
                        let (caps, dmabuf) = read_caps(&command.payload)?;
                        if dmabuf != self.dmabuf_caps {
                            return Err(G2gError::UnsupportedDomain);
                        }
                        if self.caps.as_ref() != Some(&caps) {
                            self.caps = Some(caps.clone());
                            out.push(PipelinePacket::CapsChanged(caps)).await?;
                        }
                    }
                    COMMAND_EOS => {
                        out.push(PipelinePacket::Eos).await?;
                        return Ok(sequence);
                    }
                    COMMAND_RELEASE_BUFFER => {
                        return Err(protocol_error("sink sent RELEASE_BUFFER"));
                    }
                    // gst ignores a command it does not know, the protocol may grow
                    _ => {}
                }
            }
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Unix file descriptor source",
            "Source/IPC",
            "Receives frames as memfd or dma-buf descriptors over a unix socket from a GStreamer-compatible unixfdsink",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "socket-path" | "socket-type" if self.caps.is_some() => Err(PropError::Value),
            "socket-path" => {
                self.socket_path = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "socket-type" => {
                let name = value.as_str().ok_or(PropError::Type)?;
                self.socket_type = SocketType::from_name(name).ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "socket-path" => Some(PropValue::Str(self.socket_path.clone())),
            "socket-type" => Some(PropValue::Str(self.socket_type.name().to_string())),
            _ => None,
        }
    }
}

static SRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "socket-path",
        PropKind::Str,
        "path or abstract name of the unix socket the sink serves on",
    ),
    PropertySpec::new("socket-type", PropKind::Str, "where the socket lives")
        .with_enum_values(SOCKET_TYPE_NAMES)
        .with_default(SocketType::Path.name()),
];
