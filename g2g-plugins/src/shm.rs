//! GStreamer's shared-memory IPC pair (M1081, `shm` feature): [`ShmSink`] and
//! [`ShmSrc`] speak the `shmpipe` protocol of `sys/shm/` in gst-plugins-bad, so
//! either end can be a `gst-launch-1.0` process. The wire itself is in
//! [`crate::shmpipe`].
//!
//! The sink owns a unix control socket and a shared-memory area, copies each
//! frame's bytes into a block of that area, and tells every connected client
//! where the block is. A block is reused once every client that was told about
//! it has acknowledged it. The source maps the announced area read-only, copies
//! each announced block into a fresh `Frame`, and acknowledges at once. That
//! copy is what keeps the sink's area from filling behind a slow downstream: a
//! zero-copy lend would have to hold the acknowledgement until the frame is
//! dropped, which needs a frame-lifetime hook the domain does not have.
//!
//! The protocol carries no caps, exactly as in GStreamer, where a receiver
//! spells them out with a `capsfilter`. Here the source declares them itself,
//! through `bytestream-format` for a container or `caps` for anything the
//! `capsfilter` syntax can name.
//!
//! Two deviations from the gst elements, both noted on the properties: gst's
//! `buffer-time` blocks until the oldest unacknowledged buffer is within that
//! much stream time of the new one and never drops, while here it bounds the
//! wait for space and then drops the frame; and the source retries its connect
//! for [`CONNECT_RETRY_WINDOW`] instead of failing outright, so the two halves
//! of a pipeline can start in either order.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use core::time::Duration;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use std::io::{Error as IoError, ErrorKind};
use std::os::unix::net::UnixListener as StdUnixListener;
use std::path::{Path, PathBuf};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};

use g2g_core::frame::Frame;
use g2g_core::log::io_err;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::capsfilter::parse_caps;
use crate::filesrc::{encoding_from_str, encoding_to_str};
use crate::shmpipe::{
    buffer_range, AreaAllocator, Command, MappedArea, COMMAND_BYTES, MAX_AREA_BYTES,
    MAX_AREA_NAME_BYTES,
};

/// gst `shmsink`'s `shm-size` default, 64 MiB.
const DEFAULT_SHM_SIZE: usize = 67_108_864;
/// gst `shmsink`'s `perms` default, 0640.
const DEFAULT_PERMS: u32 = 0o640;
/// The id of the one area a sink creates. gst counts from 1 as well.
const FIRST_AREA_ID: i32 = 1;
/// The container a source declares when neither `caps` nor `bytestream-format`
/// is set, matching the other byte-stream sources that cannot know their type.
const DEFAULT_ENCODING: ByteStreamEncoding = ByteStreamEncoding::MpegTs;
/// Areas a source will hold mapped at once. A sink announces a second one only
/// while resizing, so this is only a bound on a peer that keeps announcing.
const MAX_MAPPED_AREAS: usize = 8;
/// How long the source keeps retrying a connect the sink has not accepted yet,
/// so the two halves of a pipeline can start in either order.
pub const CONNECT_RETRY_WINDOW: Duration = Duration::from_secs(5);
/// Gap between those connect attempts.
const CONNECT_RETRY_GAP: Duration = Duration::from_millis(20);
/// How long the clients get to acknowledge the buffers still out when the
/// stream ends, before the area is torn down anyway.
const EOS_DRAIN_WINDOW: Duration = Duration::from_secs(2);
/// Bytes read from a client socket per syscall. Only acknowledgements come back
/// upstream, so one command's worth at a time is enough.
const ACK_READ_CHUNK: usize = COMMAND_BYTES;

/// [`DEFAULT_SHM_SIZE`] as text, since a property spec takes a `&'static str`.
const DEFAULT_SHM_SIZE_TEXT: &str = "67108864";
/// [`DEFAULT_PERMS`] as text, in the decimal gst prints.
const DEFAULT_PERMS_TEXT: &str = "416";
/// Every mode bit, the top of the 0 - 4095 range gst's `perms` declares.
const MAX_PERMS: u32 = 0o7777;
const MAX_PERMS_TEXT: &str = "4095";
/// The `shm-size` ceiling: gst's property is a `guint`, so no gst peer can name
/// a larger area either.
const MAX_SHM_SIZE_TEXT: &str = "4294967295";
/// The `buffer-time` value that means "wait as long as it takes".
const BUFFER_TIME_DISABLED_TEXT: &str = "-1";
const MAX_INT64_TEXT: &str = "9223372036854775807";
const UNLIMITED_BUFFERS_TEXT: &str = "-1";

fn protocol_error(reason: &'static str) -> G2gError {
    io_err(IoError::new(ErrorKind::InvalidData, reason))
}

/// A frame carrying `bytes`, stamped and sequenced the way the other byte
/// sources stamp a chunk.
fn byte_frame(bytes: Vec<u8>, sequence: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            arrival_ns: g2g_core::metrics::monotonic_ns(),
            ..FrameTiming::default()
        },
        sequence,
        meta: Default::default(),
    }
}

/// Bind `path`, taking over a socket file left behind by a process that is gone.
/// gst falls back to `path.0`, `path.1` and so on instead, which moves the
/// socket out from under the peer that was told `path`.
fn bind_control_socket(path: &Path) -> Result<StdUnixListener, G2gError> {
    match StdUnixListener::bind(path) {
        Ok(listener) => Ok(listener),
        Err(error) if error.kind() == ErrorKind::AddrInUse => {
            let stale = std::os::unix::net::UnixStream::connect(path).is_err();
            if !stale {
                return Err(io_err(error));
            }
            std::fs::remove_file(path).map_err(io_err)?;
            StdUnixListener::bind(path).map_err(io_err)
        }
        Err(error) => Err(io_err(error)),
    }
}

/// A client attached to the sink's control socket.
#[derive(Debug)]
struct Client {
    id: u64,
    stream: UnixStream,
    /// Acknowledgement bytes that did not complete a command yet.
    partial: Vec<u8>,
}

/// A block the sink handed out and the clients that have not acknowledged it.
#[derive(Debug)]
struct SentBuffer {
    offset: usize,
    awaiting: Vec<u64>,
}

/// The sink's half of the pipe: the control socket, the shared-memory area, the
/// allocator over it, and the buffers still in flight.
#[derive(Debug)]
struct ShmServer {
    socket_path: PathBuf,
    bound: Option<StdUnixListener>,
    listener: Option<UnixListener>,
    area: MappedArea,
    allocator: AreaAllocator,
    sent: Vec<SentBuffer>,
    clients: Vec<Client>,
    next_client_id: u64,
}

impl ShmServer {
    fn create(socket_path: &Path, shm_size: usize, perms: u32) -> Result<Self, G2gError> {
        let bound = bind_control_socket(socket_path)?;
        bound.set_nonblocking(true).map_err(io_err)?;
        std::fs::set_permissions(
            socket_path,
            std::os::unix::fs::PermissionsExt::from_mode(perms),
        )
        .map_err(io_err)?;
        let area = MappedArea::create(FIRST_AREA_ID, shm_size, perms).map_err(io_err)?;
        Ok(Self {
            socket_path: socket_path.to_path_buf(),
            bound: Some(bound),
            listener: None,
            allocator: AreaAllocator::new(area.len()),
            area,
            sent: Vec::new(),
            clients: Vec::new(),
            next_client_id: 0,
        })
    }

    /// Hand the bound socket to tokio. Only callable from inside the runtime,
    /// which is why the bind itself happens in `configure_pipeline`.
    fn adopt(&mut self) -> Result<(), G2gError> {
        if self.listener.is_none() {
            let bound = self.bound.take().ok_or(G2gError::NotConfigured)?;
            self.listener = Some(UnixListener::from_std(bound).map_err(io_err)?);
        }
        Ok(())
    }

    /// Greet a new client with the area it is to map, and keep it if that got
    /// through.
    async fn attach(&mut self, mut stream: UnixStream) {
        let mut name = Vec::from(self.area.name().as_bytes());
        name.push(0);
        let command = Command::NewShmArea {
            area_id: self.area.id(),
            size: self.area.len() as u64,
            path_size: name.len() as u32,
        };
        if stream.write_all(&command.encode()).await.is_err()
            || stream.write_all(&name).await.is_err()
        {
            return;
        }
        let id = self.next_client_id;
        self.next_client_id += 1;
        self.clients.push(Client {
            id,
            stream,
            partial: Vec::new(),
        });
    }

    /// Take every client already waiting in the listen backlog, without ever
    /// blocking on there being one.
    async fn accept_pending(&mut self) -> Result<(), G2gError> {
        self.adopt()?;
        loop {
            let listener = self.listener.as_ref().ok_or(G2gError::NotConfigured)?;
            let mut cx = Context::from_waker(Waker::noop());
            let stream = match listener.poll_accept(&mut cx) {
                Poll::Ready(Ok((stream, _peer))) => stream,
                Poll::Ready(Err(e)) => return Err(io_err(e)),
                Poll::Pending => return Ok(()),
            };
            self.attach(stream).await;
        }
    }

    /// Read every acknowledgement waiting on a client socket and free the blocks
    /// they release. A client that closed, errored, or sent something other than
    /// an acknowledgement is dropped.
    fn drain_acks(&mut self) {
        let mut index = 0;
        while index < self.clients.len() {
            if self.read_acks(index) {
                index += 1;
            } else {
                let client = self.clients.swap_remove(index);
                self.forget_client(client.id);
            }
        }
    }

    /// `false` means this client is finished with, for any reason.
    fn read_acks(&mut self, index: usize) -> bool {
        loop {
            let mut chunk = [0u8; ACK_READ_CHUNK];
            let filled = match self.clients[index].stream.try_read(&mut chunk) {
                Ok(0) => return false,
                Ok(filled) => filled,
                Err(e) if e.kind() == ErrorKind::WouldBlock => return true,
                Err(_) => return false,
            };
            self.clients[index]
                .partial
                .extend_from_slice(&chunk[..filled]);
            while self.clients[index].partial.len() >= COMMAND_BYTES {
                let raw: Vec<u8> = self.clients[index].partial.drain(..COMMAND_BYTES).collect();
                let client_id = self.clients[index].id;
                match Command::decode(&raw) {
                    Some(Command::AckBuffer { area_id, offset }) if area_id == self.area.id() => {
                        self.acknowledge(client_id, offset);
                    }
                    // gst's sp_writer_recv treats anything else on this
                    // direction as fatal for the client.
                    _ => return false,
                }
            }
        }
    }

    /// One client is done with the block at `offset`; free it once they all are.
    fn acknowledge(&mut self, client_id: u64, offset: u64) {
        let Some(index) = self.sent.iter().position(|b| b.offset as u64 == offset) else {
            return;
        };
        self.sent[index].awaiting.retain(|id| *id != client_id);
        if self.sent[index].awaiting.is_empty() {
            let buffer = self.sent.remove(index);
            self.allocator.free(buffer.offset);
        }
    }

    /// A gone client owes nothing, so its buffers free as soon as the rest have
    /// acknowledged them.
    fn forget_client(&mut self, client_id: u64) {
        let mut index = 0;
        while index < self.sent.len() {
            self.sent[index].awaiting.retain(|id| *id != client_id);
            if self.sent[index].awaiting.is_empty() {
                let buffer = self.sent.remove(index);
                self.allocator.free(buffer.offset);
            } else {
                index += 1;
            }
        }
    }

    /// Wake on the next thing that can make room: an acknowledgement from a
    /// client, or a client arriving. An accepted client is attached here, so a
    /// caller waiting for its first one gets it by waiting.
    async fn wait_for_progress(&mut self) -> Result<(), G2gError> {
        self.adopt()?;
        let accepted = {
            let Self {
                listener, clients, ..
            } = self;
            let listener = listener.as_ref().ok_or(G2gError::NotConfigured)?;
            core::future::poll_fn(|cx| {
                for client in clients.iter() {
                    if client.stream.poll_read_ready(cx).is_ready() {
                        return Poll::Ready(Ok(None));
                    }
                }
                match listener.poll_accept(cx) {
                    Poll::Ready(Ok((stream, _peer))) => Poll::Ready(Ok(Some(stream))),
                    Poll::Ready(Err(e)) => Poll::Ready(Err(io_err(e))),
                    Poll::Pending => Poll::Pending,
                }
            })
            .await?
        };
        if let Some(stream) = accepted {
            self.attach(stream).await;
        }
        Ok(())
    }

    /// Tell every client about the block at `offset`, and remember who owes an
    /// acknowledgement. Frees the block again when nobody took it.
    async fn send_buffer(&mut self, offset: usize, size: usize) {
        let command = Command::NewBuffer {
            area_id: self.area.id(),
            offset: offset as u64,
            size: size as u64,
        }
        .encode();
        let mut awaiting = Vec::new();
        let mut index = 0;
        while index < self.clients.len() {
            match self.clients[index].stream.write_all(&command).await {
                Ok(()) => {
                    awaiting.push(self.clients[index].id);
                    index += 1;
                }
                Err(_) => {
                    let client = self.clients.swap_remove(index);
                    self.forget_client(client.id);
                }
            }
        }
        if awaiting.is_empty() {
            self.allocator.free(offset);
            return;
        }
        self.sent.push(SentBuffer { offset, awaiting });
    }

    /// Let the clients acknowledge what is still out, then tell them the area is
    /// going away.
    async fn close(&mut self) {
        let deadline = tokio::time::Instant::now() + EOS_DRAIN_WINDOW;
        while !self.sent.is_empty() && !self.clients.is_empty() {
            self.drain_acks();
            if self.sent.is_empty() {
                break;
            }
            match tokio::time::timeout_at(deadline, self.wait_for_progress()).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) | Err(_) => break,
            }
        }
        let command = Command::CloseShmArea {
            area_id: self.area.id(),
        }
        .encode();
        for client in &mut self.clients {
            let _ = client.stream.write_all(&command).await;
            let _ = client.stream.shutdown().await;
        }
        self.clients.clear();
        self.sent.clear();
    }
}

impl Drop for ShmServer {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::shm::ShmSink;
///
/// // gst-launch equivalent: shmsink socket-path=/tmp/g2g-shm wait-for-connection=true
/// let sink = ShmSink::new("/tmp/g2g-shm");
/// ```
#[derive(Debug)]
pub struct ShmSink {
    socket_path: PathBuf,
    shm_size: usize,
    perms: u32,
    wait_for_connection: bool,
    /// `None` is gst's `-1`: wait for space as long as it takes.
    buffer_time_ns: Option<u64>,
    server: Option<ShmServer>,
    frames_sent: u64,
    frames_dropped: u64,
}

impl Default for ShmSink {
    fn default() -> Self {
        Self {
            socket_path: PathBuf::new(),
            shm_size: DEFAULT_SHM_SIZE,
            perms: DEFAULT_PERMS,
            wait_for_connection: true,
            buffer_time_ns: None,
            server: None,
            frames_sent: 0,
            frames_dropped: 0,
        }
    }
}

impl ShmSink {
    /// Serve the incoming frames through a control socket at `socket_path`.
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: socket_path.into(),
            ..Self::default()
        }
    }

    pub fn with_shm_size(mut self, bytes: usize) -> Self {
        self.shm_size = bytes;
        self
    }

    pub fn with_wait_for_connection(mut self, wait: bool) -> Self {
        self.wait_for_connection = wait;
        self
    }

    /// Create the control socket and the shared-memory area now, so a peer can
    /// connect before the pipeline runs. Idempotent; `configure_pipeline` calls
    /// it when nothing is open yet.
    pub fn open(&mut self) -> Result<(), G2gError> {
        if self.server.is_some() {
            return Ok(());
        }
        if self.socket_path.as_os_str().is_empty() {
            return Err(G2gError::NotConfigured);
        }
        self.server = Some(ShmServer::create(
            &self.socket_path,
            self.shm_size,
            self.perms,
        )?);
        Ok(())
    }

    /// The generated name of the shared-memory area, once it exists.
    pub fn shm_area_name(&self) -> Option<&str> {
        self.server.as_ref().map(|s| s.area.name())
    }

    /// Clients currently attached.
    pub fn client_count(&self) -> usize {
        self.server.as_ref().map_or(0, |s| s.clients.len())
    }

    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    /// Frames given up on because `buffer-time` elapsed with the area full.
    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped
    }

    /// The offset of a block for `size` bytes, or `None` when `buffer-time`
    /// elapsed with the area still full.
    async fn claim_block(&mut self, size: usize) -> Result<Option<usize>, G2gError> {
        let deadline = self
            .buffer_time_ns
            .map(|ns| tokio::time::Instant::now() + Duration::from_nanos(ns));
        let server = self.server.as_mut().ok_or(G2gError::NotConfigured)?;
        loop {
            server.accept_pending().await?;
            server.drain_acks();
            if let Some(offset) = server.allocator.alloc(size) {
                return Ok(Some(offset));
            }
            match deadline {
                None => server.wait_for_progress().await?,
                Some(deadline) => {
                    match tokio::time::timeout_at(deadline, server.wait_for_progress()).await {
                        Ok(result) => result?,
                        Err(_) => return Ok(None),
                    }
                }
            }
        }
    }

    async fn send_frame(&mut self, bytes: &[u8]) -> Result<(), G2gError> {
        {
            let server = self.server.as_mut().ok_or(G2gError::NotConfigured)?;
            server.accept_pending().await?;
            if bytes.len() > server.area.len() {
                return Err(io_err(IoError::new(
                    ErrorKind::InvalidInput,
                    "frame is larger than the whole shm area",
                )));
            }
            while self.wait_for_connection && server.clients.is_empty() {
                server.wait_for_progress().await?;
            }
        }
        let Some(offset) = self.claim_block(bytes.len()).await? else {
            self.frames_dropped += 1;
            return Ok(());
        };
        let server = self.server.as_mut().ok_or(G2gError::NotConfigured)?;
        server.area.write(offset, bytes).map_err(io_err)?;
        server.send_buffer(offset, bytes.len()).await;
        self.frames_sent += 1;
        Ok(())
    }
}

impl AsyncElement for ShmSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Copies host memory into the area, so it takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// The area carries opaque bytes, and the protocol has no field for caps.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.open()?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let bytes = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    self.send_frame(bytes).await?;
                }
                PipelinePacket::Eos => {
                    if let Some(server) = self.server.as_mut() {
                        server.close().await;
                    }
                    self.server = None;
                }
                // The protocol has no field for caps, segments or flushes, and
                // a buffer already announced stays announced.
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SINK_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Shared memory sink",
            "Sink/IPC",
            "Sends frames over POSIX shared memory to a GStreamer-compatible shmsrc",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "socket-path" => {
                self.socket_path = PathBuf::from(value.as_str().ok_or(PropError::Type)?);
                Ok(())
            }
            "shm-size" => {
                let bytes = value.as_uint().ok_or(PropError::Type)?;
                if bytes == 0 || bytes > MAX_AREA_BYTES {
                    return Err(PropError::Value);
                }
                self.shm_size = bytes as usize;
                Ok(())
            }
            "perms" => {
                let perms = value.as_uint().ok_or(PropError::Type)?;
                if perms > MAX_PERMS as u64 {
                    return Err(PropError::Value);
                }
                self.perms = perms as u32;
                Ok(())
            }
            "wait-for-connection" => {
                self.wait_for_connection = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "buffer-time" => {
                let ns = value.as_int().ok_or(PropError::Type)?;
                self.buffer_time_ns = (ns >= 0).then_some(ns as u64);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "socket-path" => Some(PropValue::Str(
                self.socket_path.to_string_lossy().into_owned(),
            )),
            "shm-size" => Some(PropValue::Uint(self.shm_size as u64)),
            "perms" => Some(PropValue::Uint(self.perms as u64)),
            "wait-for-connection" => Some(PropValue::Bool(self.wait_for_connection)),
            "buffer-time" => Some(PropValue::Int(
                self.buffer_time_ns.map_or(-1, |ns| ns as i64),
            )),
            "shm-area-name" => Some(PropValue::Str(
                self.shm_area_name().unwrap_or_default().to_string(),
            )),
            _ => None,
        }
    }
}

static SINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "socket-path",
        PropKind::Str,
        "path of the unix control socket the clients connect to",
    ),
    PropertySpec::new(
        "shm-size",
        PropKind::Uint,
        "bytes in the shared memory area the buffers are allocated from",
    )
    .with_default(DEFAULT_SHM_SIZE_TEXT)
    .with_range("1", MAX_SHM_SIZE_TEXT),
    PropertySpec::new(
        "perms",
        PropKind::Uint,
        "mode bits set on the shm area and the control socket",
    )
    .with_default(DEFAULT_PERMS_TEXT)
    .with_range("0", MAX_PERMS_TEXT),
    PropertySpec::new(
        "wait-for-connection",
        PropKind::Bool,
        "hold the stream until a client connects",
    )
    .with_default("true"),
    PropertySpec::new(
        "buffer-time",
        PropKind::Int,
        "nanoseconds to wait for room in the area before dropping a frame (-1 = wait)",
    )
    .with_default(BUFFER_TIME_DISABLED_TEXT)
    .with_range(BUFFER_TIME_DISABLED_TEXT, MAX_INT64_TEXT),
    PropertySpec::new(
        "shm-area-name",
        PropKind::Str,
        "generated name of the shared memory area, empty until it is created",
    )
    .read_only(),
];

impl PadTemplates for ShmSink {
    /// Wildcard sink, matching the runtime `AcceptsAny` constraint.
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::shm::ShmSrc;
///
/// // gst-launch equivalent: shmsrc socket-path=/tmp/g2g-shm
/// let source = ShmSrc::new("/tmp/g2g-shm");
/// ```
#[derive(Debug)]
pub struct ShmSrc {
    socket_path: PathBuf,
    caps: Caps,
    /// The `caps` property text, kept so `get_property` round-trips it.
    caps_text: String,
    frame_limit: u64,
    area_name: Option<String>,
    configured: bool,
}

impl Default for ShmSrc {
    fn default() -> Self {
        Self {
            socket_path: PathBuf::new(),
            caps: Caps::ByteStream {
                encoding: DEFAULT_ENCODING,
            },
            caps_text: String::new(),
            frame_limit: u64::MAX,
            area_name: None,
            configured: false,
        }
    }
}

impl ShmSrc {
    /// Read the frames a sink serves on the control socket at `socket_path`.
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: socket_path.into(),
            ..Self::default()
        }
    }

    /// Declare what the bytes are, for the receiver of a producer that cannot
    /// say so on the wire.
    pub fn with_caps(mut self, caps: Caps) -> Self {
        self.caps = caps;
        self
    }

    /// The name of the area the sink announced, once it has.
    pub fn shm_area_name(&self) -> Option<&str> {
        self.area_name.as_deref()
    }

    async fn connect(&self) -> Result<UnixStream, G2gError> {
        let deadline = tokio::time::Instant::now() + CONNECT_RETRY_WINDOW;
        loop {
            match UnixStream::connect(&self.socket_path).await {
                Ok(stream) => return Ok(stream),
                Err(error) if tokio::time::Instant::now() >= deadline => return Err(io_err(error)),
                Err(_) => tokio::time::sleep(CONNECT_RETRY_GAP).await,
            }
        }
    }

    /// Read the area name that follows a `NewShmArea` command and map it.
    async fn map_area(
        &mut self,
        stream: &mut UnixStream,
        area_id: i32,
        size: u64,
        path_size: u32,
    ) -> Result<MappedArea, G2gError> {
        if path_size == 0 || path_size as usize > MAX_AREA_NAME_BYTES {
            return Err(protocol_error("shm area name length is out of range"));
        }
        if size == 0 || size > MAX_AREA_BYTES {
            return Err(protocol_error("shm area size is out of range"));
        }
        let mut raw = alloc::vec![0u8; path_size as usize];
        stream.read_exact(&mut raw).await.map_err(io_err)?;
        let end = raw.iter().position(|b| *b == 0).unwrap_or(raw.len());
        let name = core::str::from_utf8(&raw[..end])
            .map_err(|_| protocol_error("shm area name is not utf-8"))?;
        self.area_name = Some(name.to_string());
        MappedArea::open_readonly(area_id, name, size as usize).map_err(io_err)
    }
}

impl SourceLoop for ShmSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps.clone()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(
            self.caps.clone(),
        ))))
    }

    fn configured_output_caps(&self) -> Option<Caps> {
        Some(self.caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.socket_path.as_os_str().is_empty() {
            return Err(G2gError::NotConfigured);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            if crate::numbuffers::finished_at_zero_limit(self.frame_limit, out).await? {
                return Ok(0);
            }
            let mut stream = self.connect().await?;
            let mut areas: Vec<MappedArea> = Vec::new();
            let mut sequence = 0u64;
            while sequence < self.frame_limit {
                let mut raw = [0u8; COMMAND_BYTES];
                match stream.read_exact(&mut raw).await {
                    Ok(_) => {}
                    Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(io_err(e)),
                }
                let command = Command::decode(&raw)
                    .ok_or_else(|| protocol_error("unknown shmpipe command"))?;
                match command {
                    Command::NewShmArea {
                        area_id,
                        size,
                        path_size,
                    } => {
                        if areas.len() >= MAX_MAPPED_AREAS {
                            return Err(protocol_error("peer announced too many shm areas"));
                        }
                        let area = self.map_area(&mut stream, area_id, size, path_size).await?;
                        areas.push(area);
                    }
                    Command::CloseShmArea { area_id } => areas.retain(|a| a.id() != area_id),
                    Command::NewBuffer {
                        area_id,
                        offset,
                        size,
                    } => {
                        let area = areas
                            .iter()
                            .find(|a| a.id() == area_id)
                            .ok_or_else(|| protocol_error("buffer names an unmapped shm area"))?;
                        let range = buffer_range(area.len(), offset, size)
                            .ok_or_else(|| protocol_error("buffer lies outside the shm area"))?;
                        let bytes = Vec::from(area.read(range));
                        // The copy above is what lets this acknowledgement go out
                        // now instead of when the frame is dropped downstream.
                        stream
                            .write_all(&Command::AckBuffer { area_id, offset }.encode())
                            .await
                            .map_err(io_err)?;
                        out.push(PipelinePacket::DataFrame(byte_frame(bytes, sequence)))
                            .await?;
                        sequence += 1;
                    }
                    // Only a client sends this, and this is the client.
                    Command::AckBuffer { .. } => {
                        return Err(protocol_error("sink sent an acknowledgement"))
                    }
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Shared memory source",
            "Source/IPC",
            "Reads frames a GStreamer-compatible shmsink serves over POSIX shared memory",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "socket-path" => {
                self.socket_path = PathBuf::from(value.as_str().ok_or(PropError::Type)?);
                Ok(())
            }
            "bytestream-format" => {
                let text = value.as_str().ok_or(PropError::Type)?;
                let encoding = encoding_from_str(text).ok_or(PropError::Value)?;
                self.caps = Caps::ByteStream { encoding };
                self.caps_text = String::new();
                Ok(())
            }
            "caps" => {
                let text = value.as_str().ok_or(PropError::Type)?;
                self.caps = parse_caps(text).ok_or(PropError::Value)?;
                self.caps_text = text.to_string();
                Ok(())
            }
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.frame_limit, &value),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "socket-path" => Some(PropValue::Str(
                self.socket_path.to_string_lossy().into_owned(),
            )),
            "bytestream-format" => match &self.caps {
                Caps::ByteStream { encoding } => {
                    Some(PropValue::Str(encoding_to_str(*encoding).to_string()))
                }
                _ => None,
            },
            "caps" if !self.caps_text.is_empty() => Some(PropValue::Str(self.caps_text.clone())),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.frame_limit)),
            "shm-area-name" => Some(PropValue::Str(
                self.shm_area_name().unwrap_or_default().to_string(),
            )),
            _ => None,
        }
    }
}

static SRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "socket-path",
        PropKind::Str,
        "path of the unix control socket the sink serves on",
    ),
    PropertySpec::new(
        "bytestream-format",
        PropKind::Str,
        "container of the byte stream: mpegts | matroska | ogg | flv | mp4",
    )
    .with_default("mpegts"),
    PropertySpec::new(
        "caps",
        PropKind::Str,
        "caps to declare instead, gst-launch syntax: e.g. video/x-raw,format=i420,width=320,height=240",
    ),
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "frames to emit then EOS (-1 = until the sink closes)",
    )
    .with_default(UNLIMITED_BUFFERS_TEXT)
    .with_range(UNLIMITED_BUFFERS_TEXT, MAX_INT64_TEXT),
    PropertySpec::new(
        "shm-area-name",
        PropKind::Str,
        "name of the shared memory area the sink announced, empty until it does",
    )
    .read_only(),
];
