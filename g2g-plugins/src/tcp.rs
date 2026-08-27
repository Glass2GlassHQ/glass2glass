//! Plain TCP byte-stream elements (M1068, `tcp` feature): the four elements of
//! GStreamer's `tcp` plugin. `TcpServerSrc` / `TcpClientSrc` read a socket and
//! emit its bytes as `DataFrame` chunks exactly the way [`FileSrc`](crate::filesrc)
//! reads a file, so `tcpclientsrc ! typefind ! decodebin` types a stream the same
//! way `filesrc ! typefind ! decodebin` does. `TcpServerSink` / `TcpClientSink`
//! write every incoming `DataFrame`'s bytes back out to the socket.
//!
//! These carry raw bytes with no framing of their own. The
//! [`remote`](crate::remotesrc) pair also speaks TCP, but it carries serialized
//! `PipelinePacket`s (caps, timing, segments) behind a length prefix; here the
//! wire holds nothing but the payload, which is what makes these interoperable
//! with anything that speaks a socket.
//!
//! A byte stream carries no caps, so the sources declare a container through
//! `bytestream-format` (default `mpegts`, matching the other network byte-stream
//! sources). There is no `auto` sniff: nothing can be read before the peer
//! connects, and a downstream `typefind` re-declares the type from the content
//! anyway.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use std::io::{Error as IoError, ErrorKind};
use std::net::{
    SocketAddr, TcpListener as StdTcpListener, TcpStream as StdTcpStream, ToSocketAddrs,
};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

use crate::bytestream::byte_frame;
use crate::filesink::io_err;
use crate::filesrc::{encoding_from_str, encoding_to_str};

/// The `port` every element of the GStreamer `tcp` plugin defaults to.
const DEFAULT_PORT: u16 = 4953;
/// The `host` every element of the GStreamer `tcp` plugin defaults to.
const DEFAULT_HOST: &str = "localhost";
/// Bytes per socket read, matching the gst `tcpserversrc` / `tcpclientsrc`
/// `blocksize` default.
const DEFAULT_BLOCKSIZE: usize = 4096;
/// Largest `blocksize` a property may ask for, so a bad value cannot demand a
/// huge per-read allocation.
const MAX_BLOCKSIZE: u64 = 1 << 30;
/// The container a source declares when `bytestream-format` is not set, matching
/// `srtsrc` (the other byte-stream network source that cannot know its type).
const DEFAULT_ENCODING: ByteStreamEncoding = ByteStreamEncoding::MpegTs;

/// [`DEFAULT_PORT`] as text, since a property spec takes a `&'static str`.
const DEFAULT_PORT_TEXT: &str = "4953";
/// [`DEFAULT_BLOCKSIZE`] as text.
const DEFAULT_BLOCKSIZE_TEXT: &str = "4096";
/// [`MAX_BLOCKSIZE`] as text.
const MAX_BLOCKSIZE_TEXT: &str = "1073741824";
const PORT_MIN_TEXT: &str = "0";
const PORT_MAX_TEXT: &str = "65535";
/// The `num-buffers` value that means "run until the peer closes".
const UNLIMITED_BUFFERS_TEXT: &str = "-1";
const MAX_BUFFERS_TEXT: &str = "9223372036854775807";

/// The `host` spec. gst words the blurb differently for a listener and a dialer,
/// so the element supplies it.
const fn host_spec(blurb: &'static str) -> PropertySpec {
    PropertySpec::new("host", PropKind::Str, blurb).with_default(DEFAULT_HOST)
}

/// The `port` spec, blurb supplied for the same reason as [`host_spec`]'s.
const fn port_spec(blurb: &'static str) -> PropertySpec {
    PropertySpec::new("port", PropKind::Uint, blurb)
        .with_default(DEFAULT_PORT_TEXT)
        .with_range(PORT_MIN_TEXT, PORT_MAX_TEXT)
}

/// The port a `port=0` bind actually got. Both server elements report it, and
/// neither lets it be set.
const CURRENT_PORT_SPEC: PropertySpec = PropertySpec::new(
    "current-port",
    PropKind::Uint,
    "the port the socket is actually bound to (port=0 picks one)",
)
.with_range(PORT_MIN_TEXT, PORT_MAX_TEXT)
.read_only();

/// The read size both sources take.
const BLOCKSIZE_SPEC: PropertySpec = PropertySpec::new(
    "blocksize",
    PropKind::Uint,
    "bytes per socket read, and per emitted DataFrame",
)
.with_default(DEFAULT_BLOCKSIZE_TEXT)
.with_range("1", MAX_BLOCKSIZE_TEXT);

/// The run length both sources take.
const NUM_BUFFERS_SPEC: PropertySpec = PropertySpec::new(
    "num-buffers",
    PropKind::Int,
    "chunks to emit then EOS (-1 = until the peer closes)",
)
.with_default(UNLIMITED_BUFFERS_TEXT)
.with_range(UNLIMITED_BUFFERS_TEXT, MAX_BUFFERS_TEXT);

/// The container both sources declare for the bytes they carry.
const FORMAT_SPEC: PropertySpec = PropertySpec::new(
    "bytestream-format",
    PropKind::Str,
    "container of the byte stream: mpegts | matroska | ogg | flv | mp4",
)
.with_default("mpegts");

/// Where an element reaches its peer. `host` is a name or a literal address (gst
/// defaults it to `localhost`), so it is resolved at bind / connect time instead
/// of being parsed into a `SocketAddr` when the property is set.
#[derive(Debug, Clone)]
struct Endpoint {
    host: String,
    port: u16,
}

impl Default for Endpoint {
    fn default() -> Self {
        Self {
            host: DEFAULT_HOST.to_string(),
            port: DEFAULT_PORT,
        }
    }
}

impl Endpoint {
    fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
        }
    }

    fn resolve(&self) -> Result<SocketAddr, G2gError> {
        (self.host.as_str(), self.port)
            .to_socket_addrs()
            .map_err(io_err)?
            .next()
            .ok_or_else(|| {
                io_err(IoError::new(
                    ErrorKind::AddrNotAvailable,
                    "host resolved to no address",
                ))
            })
    }

    /// `Some` when `name` is `host` or `port`, so the caller returns the result;
    /// `None` when it is one of the element's own property names.
    fn set_property(&mut self, name: &str, value: &PropValue) -> Option<Result<(), PropError>> {
        match name {
            "host" => Some(match value.as_str() {
                Some(host) => {
                    self.host = host.to_string();
                    Ok(())
                }
                None => Err(PropError::Type),
            }),
            "port" => Some(match value.as_uint() {
                Some(port) if port <= u16::MAX as u64 => {
                    self.port = port as u16;
                    Ok(())
                }
                Some(_) => Err(PropError::Value),
                None => Err(PropError::Type),
            }),
            _ => None,
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "host" => Some(PropValue::Str(self.host.clone())),
            "port" => Some(PropValue::Uint(self.port as u64)),
            _ => None,
        }
    }
}

/// The listening socket the two server elements share. It is bound
/// synchronously, outside any runtime, so `current-port` reads back the real
/// port before the pipeline runs; tokio adopts it on first use, which has to
/// happen inside the runtime.
#[derive(Debug, Default)]
struct BoundListener {
    bound: Option<StdTcpListener>,
    adopted: Option<TcpListener>,
}

impl BoundListener {
    fn bind(&mut self, endpoint: &Endpoint) -> Result<u16, G2gError> {
        if self.bound.is_none() && self.adopted.is_none() {
            let listener = StdTcpListener::bind(endpoint.resolve()?).map_err(io_err)?;
            listener.set_nonblocking(true).map_err(io_err)?;
            self.bound = Some(listener);
        }
        self.port().ok_or_else(|| {
            io_err(IoError::new(
                ErrorKind::NotConnected,
                "listener has no port",
            ))
        })
    }

    fn port(&self) -> Option<u16> {
        match (&self.adopted, &self.bound) {
            (Some(l), _) => l.local_addr().ok().map(|a| a.port()),
            (None, Some(l)) => l.local_addr().ok().map(|a| a.port()),
            (None, None) => None,
        }
    }

    fn adopt(&mut self) -> Result<&TcpListener, G2gError> {
        if self.adopted.is_none() {
            let bound = self.bound.take().ok_or(G2gError::NotConfigured)?;
            self.adopted = Some(TcpListener::from_std(bound).map_err(io_err)?);
        }
        self.adopted.as_ref().ok_or(G2gError::NotConfigured)
    }
}

/// Read `stream` until the peer closes it (or `limit` chunks were emitted), then
/// end on `Eos`. Unlike the file read, a short read is emitted as it stands:
/// waiting to fill the block would hold arrived bytes back behind bytes that are
/// still on the network.
async fn read_until_close(
    stream: &mut TcpStream,
    blocksize: usize,
    limit: u64,
    out: &mut dyn OutputSink,
) -> Result<u64, G2gError> {
    let mut sequence = 0u64;
    while sequence < limit {
        let mut buf = alloc::vec![0u8; blocksize];
        let filled = stream.read(&mut buf).await.map_err(io_err)?;
        if filled == 0 {
            break;
        }
        buf.truncate(filled);
        let frame = byte_frame(buf, sequence);
        sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
    }
    out.push(PipelinePacket::Eos).await?;
    Ok(sequence)
}

/// What both TCP sources hold: the peer, the container they declare for the
/// otherwise untyped byte stream, and the read / emit limits.
#[derive(Debug)]
struct SourceSettings {
    endpoint: Endpoint,
    caps: Caps,
    blocksize: usize,
    frame_limit: u64,
    configured: bool,
}

impl Default for SourceSettings {
    fn default() -> Self {
        Self {
            endpoint: Endpoint::default(),
            caps: Caps::ByteStream {
                encoding: DEFAULT_ENCODING,
            },
            blocksize: DEFAULT_BLOCKSIZE,
            frame_limit: u64::MAX,
            configured: false,
        }
    }
}

impl SourceSettings {
    fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            endpoint: Endpoint::new(host, port),
            ..Self::default()
        }
    }

    fn set_property(&mut self, name: &str, value: &PropValue) -> Option<Result<(), PropError>> {
        if let Some(result) = self.endpoint.set_property(name, value) {
            return Some(result);
        }
        match name {
            "blocksize" => Some(match value.as_uint() {
                None => Err(PropError::Type),
                Some(0) => Err(PropError::Value),
                Some(bytes) => {
                    self.blocksize = bytes.min(MAX_BLOCKSIZE) as usize;
                    Ok(())
                }
            }),
            "num-buffers" => Some(crate::numbuffers::set_num_buffers(
                &mut self.frame_limit,
                value,
            )),
            "bytestream-format" => Some(match value.as_str() {
                Some(text) => match encoding_from_str(text) {
                    Some(encoding) => {
                        self.caps = Caps::ByteStream { encoding };
                        Ok(())
                    }
                    None => Err(PropError::Value),
                },
                None => Err(PropError::Type),
            }),
            _ => None,
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(value) = self.endpoint.get_property(name) {
            return Some(value);
        }
        match name {
            "blocksize" => Some(PropValue::Uint(self.blocksize as u64)),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.frame_limit)),
            "bytestream-format" => match &self.caps {
                Caps::ByteStream { encoding } => {
                    Some(PropValue::Str(encoding_to_str(*encoding).to_string()))
                }
                _ => None,
            },
            _ => None,
        }
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::tcp::TcpServerSrc;
///
/// // gst-launch equivalent: tcpserversrc host=0.0.0.0 port=5000
/// let source = TcpServerSrc::new("0.0.0.0", 5000);
/// ```
#[derive(Debug, Default)]
pub struct TcpServerSrc {
    settings: SourceSettings,
    listener: BoundListener,
}

impl TcpServerSrc {
    /// Listen on `host:port` and read the one client that connects.
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            settings: SourceSettings::new(host, port),
            listener: BoundListener::default(),
        }
    }

    /// Bind now and report the bound port, so a caller that asked for `port=0`
    /// can tell the peer where to connect before the pipeline runs. Idempotent;
    /// `configure_pipeline` calls it when nothing bound yet.
    pub fn bind(&mut self) -> Result<u16, G2gError> {
        self.listener.bind(&self.settings.endpoint)
    }

    /// The bound port, or `None` before the socket exists.
    pub fn current_port(&self) -> Option<u16> {
        self.listener.port()
    }
}

impl SourceLoop for TcpServerSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.settings.caps.clone()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(
            self.settings.caps.clone(),
        ))))
    }

    fn configured_output_caps(&self) -> Option<Caps> {
        Some(self.settings.caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.bind()?;
        self.settings.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.settings.configured {
                return Err(G2gError::NotConfigured);
            }
            if crate::numbuffers::finished_at_zero_limit(self.settings.frame_limit, out).await? {
                return Ok(0);
            }
            let listener = self.listener.adopt()?;
            let (mut stream, _peer) = listener.accept().await.map_err(io_err)?;
            read_until_close(
                &mut stream,
                self.settings.blocksize,
                self.settings.frame_limit,
                out,
            )
            .await
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SERVER_SRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "TCP server source",
            "Source/Network",
            "Accepts one TCP client and reads its byte stream",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.settings
            .set_property(name, &value)
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if name == "current-port" {
            return Some(PropValue::Uint(self.current_port().unwrap_or(0) as u64));
        }
        self.settings.get_property(name)
    }
}

static SERVER_SRC_PROPS: &[PropertySpec] = &[
    host_spec("address to listen on"),
    port_spec("port to listen on (0 = pick one)"),
    CURRENT_PORT_SPEC,
    BLOCKSIZE_SPEC,
    NUM_BUFFERS_SPEC,
    FORMAT_SPEC,
];

/// # Example
///
/// ```no_run
/// use g2g_plugins::tcp::TcpClientSrc;
///
/// // gst-launch equivalent: tcpclientsrc host=192.168.1.10 port=5000
/// let source = TcpClientSrc::new("192.168.1.10", 5000);
/// ```
#[derive(Debug, Default)]
pub struct TcpClientSrc {
    settings: SourceSettings,
}

impl TcpClientSrc {
    /// Connect to `host:port` and read until the peer closes.
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            settings: SourceSettings::new(host, port),
        }
    }
}

impl SourceLoop for TcpClientSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.settings.caps.clone()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(
            self.settings.caps.clone(),
        ))))
    }

    fn configured_output_caps(&self) -> Option<Caps> {
        Some(self.settings.caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.settings.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// Dials in `run`, the way `FileSrc` opens its file there: nothing before the
    /// pipeline starts needs the socket, and a server that is still coming up has
    /// until then to bind.
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.settings.configured {
                return Err(G2gError::NotConfigured);
            }
            if crate::numbuffers::finished_at_zero_limit(self.settings.frame_limit, out).await? {
                return Ok(0);
            }
            let addr = self.settings.endpoint.resolve()?;
            let mut stream = TcpStream::connect(addr).await.map_err(io_err)?;
            read_until_close(
                &mut stream,
                self.settings.blocksize,
                self.settings.frame_limit,
                out,
            )
            .await
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        CLIENT_SRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "TCP client source",
            "Source/Network",
            "Connects to a TCP server and reads its byte stream",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.settings
            .set_property(name, &value)
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.settings.get_property(name)
    }
}

static CLIENT_SRC_PROPS: &[PropertySpec] = &[
    host_spec("address of the server to read from"),
    port_spec("port of the server to read from"),
    BLOCKSIZE_SPEC,
    NUM_BUFFERS_SPEC,
    FORMAT_SPEC,
];

/// # Example
///
/// ```no_run
/// use g2g_plugins::tcp::TcpServerSink;
///
/// // gst-launch equivalent: tcpserversink host=0.0.0.0 port=5000
/// let sink = TcpServerSink::new("0.0.0.0", 5000);
/// ```
#[derive(Debug)]
pub struct TcpServerSink {
    endpoint: Endpoint,
    listener: BoundListener,
    clients: Vec<TcpStream>,
    /// Hold the first frame until a client is there to receive it, so a recorded
    /// stream starts at its first byte. With it off, frames written while nobody
    /// is connected are discarded, and a client joining later starts mid-stream.
    wait_for_connection: bool,
    bytes_written: u64,
    configured: bool,
}

impl Default for TcpServerSink {
    fn default() -> Self {
        Self {
            endpoint: Endpoint::default(),
            listener: BoundListener::default(),
            clients: Vec::new(),
            wait_for_connection: true,
            bytes_written: 0,
            configured: false,
        }
    }
}

impl TcpServerSink {
    /// Serve the byte stream to every client that connects to `host:port`.
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            endpoint: Endpoint::new(host, port),
            ..Self::default()
        }
    }

    /// Bind now and report the bound port, so a caller that asked for `port=0`
    /// can tell its clients where to connect before the pipeline runs.
    /// Idempotent; `configure_pipeline` calls it when nothing bound yet.
    pub fn bind(&mut self) -> Result<u16, G2gError> {
        self.listener.bind(&self.endpoint)
    }

    /// The bound port, or `None` before the socket exists.
    pub fn current_port(&self) -> Option<u16> {
        self.listener.port()
    }

    /// Clients currently attached.
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Take every client already waiting in the listen backlog. Never blocks:
    /// `process` has to keep serving the clients it already has.
    fn take_backlog(&mut self) -> Result<(), G2gError> {
        let listener = self.listener.adopt()?;
        let mut cx = Context::from_waker(Waker::noop());
        loop {
            match listener.poll_accept(&mut cx) {
                Poll::Ready(Ok((stream, _peer))) => self.clients.push(stream),
                Poll::Ready(Err(e)) => return Err(io_err(e)),
                Poll::Pending => return Ok(()),
            }
        }
    }

    /// Write `bytes` to every client, dropping the ones the write failed on. No
    /// per-client queue: a slow client slows the pipeline instead of growing a
    /// backlog nothing bounds.
    async fn broadcast(&mut self, bytes: &[u8]) {
        let mut index = 0;
        while index < self.clients.len() {
            match self.clients[index].write_all(bytes).await {
                Ok(()) => index += 1,
                Err(_) => {
                    self.clients.swap_remove(index);
                }
            }
        }
    }
}

impl AsyncElement for TcpServerSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Writes host memory, so it takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// A raw byte stream can carry anything.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.bind()?;
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
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    self.take_backlog()?;
                    if self.clients.is_empty() && self.wait_for_connection {
                        let listener = self.listener.adopt()?;
                        let (stream, _peer) = listener.accept().await.map_err(io_err)?;
                        self.clients.push(stream);
                    }
                    let bytes = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    self.broadcast(bytes).await;
                    self.bytes_written += bytes.len() as u64;
                }
                PipelinePacket::Eos => {
                    for client in &mut self.clients {
                        let _ = client.shutdown().await;
                    }
                    self.clients.clear();
                }
                // Caps, segments and flushes are not representable in a raw byte
                // stream, and already-sent bytes stay sent.
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SERVER_SINK_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "TCP server sink",
            "Sink/Network",
            "Serves the incoming byte stream to every connected TCP client",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(result) = self.endpoint.set_property(name, &value) {
            return result;
        }
        match name {
            "wait-for-connection" => {
                self.wait_for_connection = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(value) = self.endpoint.get_property(name) {
            return Some(value);
        }
        match name {
            "current-port" => Some(PropValue::Uint(self.current_port().unwrap_or(0) as u64)),
            "wait-for-connection" => Some(PropValue::Bool(self.wait_for_connection)),
            _ => None,
        }
    }
}

static SERVER_SINK_PROPS: &[PropertySpec] = &[
    host_spec("address to listen on"),
    port_spec("port to listen on (0 = pick one)"),
    CURRENT_PORT_SPEC,
    PropertySpec::new(
        "wait-for-connection",
        PropKind::Bool,
        "hold the stream until a client connects",
    )
    .with_default("true"),
];

impl PadTemplates for TcpServerSink {
    /// Wildcard sink, matching the runtime `AcceptsAny` constraint.
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::tcp::TcpClientSink;
///
/// // gst-launch equivalent: tcpclientsink host=192.168.1.10 port=5000
/// let sink = TcpClientSink::new("192.168.1.10", 5000);
/// ```
#[derive(Debug, Default)]
pub struct TcpClientSink {
    endpoint: Endpoint,
    /// Connected synchronously in `configure_pipeline` (no runtime needed),
    /// adopted by tokio on the first write.
    connected: Option<StdTcpStream>,
    stream: Option<TcpStream>,
    bytes_written: u64,
    eos_seen: bool,
}

impl TcpClientSink {
    /// Send the byte stream to the server at `host:port`.
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            endpoint: Endpoint::new(host, port),
            ..Self::default()
        }
    }

    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    pub fn eos_seen(&self) -> bool {
        self.eos_seen
    }

    fn socket(&mut self) -> Result<&mut TcpStream, G2gError> {
        if self.stream.is_none() {
            let connected = self.connected.take().ok_or(G2gError::NotConfigured)?;
            self.stream = Some(TcpStream::from_std(connected).map_err(io_err)?);
        }
        self.stream.as_mut().ok_or(G2gError::NotConfigured)
    }
}

impl AsyncElement for TcpClientSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Writes host memory, so it takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// A raw byte stream can carry anything.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    /// Dials here rather than on the first frame, so a server that is not there
    /// fails pipeline setup instead of surfacing as a write error after the graph
    /// has been running. A re-negotiation keeps the open socket.
    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.connected.is_none() && self.stream.is_none() {
            let stream = StdTcpStream::connect(self.endpoint.resolve()?).map_err(io_err)?;
            stream.set_nonblocking(true).map_err(io_err)?;
            self.connected = Some(stream);
        }
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
                    self.socket()?.write_all(bytes).await.map_err(io_err)?;
                    self.bytes_written += bytes.len() as u64;
                }
                PipelinePacket::Eos => {
                    // The peer reads until close, so the shutdown is what ends its
                    // stream.
                    self.socket()?.shutdown().await.map_err(io_err)?;
                    self.eos_seen = true;
                }
                // Caps, segments and flushes are not representable in a raw byte
                // stream, and already-sent bytes stay sent.
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        CLIENT_SINK_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "TCP client sink",
            "Sink/Network",
            "Connects to a TCP server and sends the incoming byte stream",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.endpoint
            .set_property(name, &value)
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.endpoint.get_property(name)
    }
}

static CLIENT_SINK_PROPS: &[PropertySpec] = &[
    host_spec("address of the server to send to"),
    port_spec("port of the server to send to"),
];

impl PadTemplates for TcpClientSink {
    /// Wildcard sink, matching the runtime `AcceptsAny` constraint.
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}
