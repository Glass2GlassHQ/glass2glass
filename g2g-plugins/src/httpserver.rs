use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::net::TcpListener as StdTcpListener;
use std::sync::Mutex;

use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

use crate::filesink::io_err;
use crate::hlssink::{fragment_info, track_timescales};
use crate::httprequest::{read_request, AbortOnDrop, HttpRequest};
use crate::mp4box::boxes;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Status {
    Ok,
    BadRequest,
    NotFound,
    MethodNotAllowed,
}

impl Status {
    fn code_and_reason(self) -> (u16, &'static str) {
        match self {
            Status::Ok => (200, "OK"),
            Status::BadRequest => (400, "Bad Request"),
            Status::NotFound => (404, "Not Found"),
            Status::MethodNotAllowed => (405, "Method Not Allowed"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BodyLength {
    Exact(usize),
    // a live stream of unknown length, ended by a zero-length chunk
    Chunked,
}

pub(crate) fn response_head(
    status: Status,
    content_type: Option<&str>,
    body_length: BodyLength,
) -> Vec<u8> {
    let (code, reason) = status.code_and_reason();
    let mut head = alloc::format!(
        "HTTP/1.1 {code} {reason}\r\nCache-Control: no-cache\r\n\
         Access-Control-Allow-Origin: *\r\nConnection: close\r\n"
    );
    if let Some(content_type) = content_type {
        head.push_str(&alloc::format!("Content-Type: {content_type}\r\n"));
    }
    match body_length {
        BodyLength::Exact(length) => head.push_str(&alloc::format!("Content-Length: {length}\r\n")),
        BodyLength::Chunked => head.push_str("Transfer-Encoding: chunked\r\n"),
    }
    head.push_str("\r\n");
    head.into_bytes()
}

// binds at configure, so a taken port fails the run before any frame flows
pub(crate) fn bind(address: &str, port: u16) -> Result<StdTcpListener, G2gError> {
    let listener = StdTcpListener::bind(crate::tcp::Endpoint::new(address, port).resolve()?)
        .map_err(io_err)?;
    listener.set_nonblocking(true).map_err(io_err)?;
    Ok(listener)
}

// the last path segment without its query, so `playlist-root` prefixes still match
pub(crate) fn requested_name(path: &str) -> &str {
    let path = path.split(['?', '#']).next().unwrap_or(path);
    path.rsplit('/').next().unwrap_or(path)
}

#[derive(Debug, Clone)]
struct PublishedFile {
    path: String,
    content_type: &'static str,
}

type PublishedFiles = Arc<Mutex<BTreeMap<String, PublishedFile>>>;

// answers GET and HEAD for the files published under their names, nothing else
#[derive(Debug)]
pub(crate) struct FileServer {
    port: u16,
    listener: Option<StdTcpListener>,
    files: PublishedFiles,
    task: Option<AbortOnDrop>,
}

impl FileServer {
    pub(crate) fn bind(address: &str, port: u16) -> Result<Self, G2gError> {
        let listener = bind(address, port)?;
        let port = listener.local_addr().map_err(io_err)?.port();
        Ok(Self {
            port,
            listener: Some(listener),
            files: PublishedFiles::default(),
            task: None,
        })
    }

    pub(crate) fn port(&self) -> u16 {
        self.port
    }

    pub(crate) fn publish(&self, name: &str, path: &str, content_type: &'static str) {
        let file = PublishedFile {
            path: path.into(),
            content_type,
        };
        self.published().insert(name.into(), file);
    }

    pub(crate) fn withdraw(&self, name: &str) {
        self.published().remove(name);
    }

    fn published(&self) -> std::sync::MutexGuard<'_, BTreeMap<String, PublishedFile>> {
        self.files.lock().expect("published file map lock")
    }

    // needs the tokio runtime, so it runs from `process` rather than configure
    pub(crate) fn start(&mut self) -> Result<(), G2gError> {
        let Some(listener) = self.listener.take() else {
            return Ok(());
        };
        let listener = TcpListener::from_std(listener).map_err(io_err)?;
        self.task = Some(AbortOnDrop(tokio::spawn(accept_file_requests(
            listener,
            self.files.clone(),
        ))));
        Ok(())
    }
}

async fn accept_file_requests(listener: TcpListener, files: PublishedFiles) {
    while let Ok((stream, _peer)) = listener.accept().await {
        tokio::spawn(answer_file_request(stream, files.clone()));
    }
}

async fn answer_file_request(mut stream: TcpStream, files: PublishedFiles) {
    let response = match read_request(&mut stream).await {
        Err(_) => response_head(Status::BadRequest, None, BodyLength::Exact(0)),
        Ok(request) => file_response(&request, &files).await,
    };
    // the client hung up, nothing left to tell it
    let _ = stream.write_all(&response).await;
}

async fn file_response(request: &HttpRequest, files: &PublishedFiles) -> Vec<u8> {
    let head_only = request.method == "HEAD";
    if request.method != "GET" && !head_only {
        return response_head(Status::MethodNotAllowed, None, BodyLength::Exact(0));
    }
    let name = requested_name(&request.path);
    let file = files
        .lock()
        .expect("published file map lock")
        .get(name)
        .cloned();
    let Some(file) = file else {
        return response_head(Status::NotFound, None, BodyLength::Exact(0));
    };
    // pruned from disk between the lookup and the read
    let Ok(body) = tokio::fs::read(&file.path).await else {
        return response_head(Status::NotFound, None, BodyLength::Exact(0));
    };
    let mut response = response_head(
        Status::Ok,
        Some(file.content_type),
        BodyLength::Exact(body.len()),
    );
    if !head_only {
        response.extend_from_slice(&body);
    }
    response
}

const DEFAULT_HOST: &str = "localhost";
const LAST_CHUNK: &[u8] = b"0\r\n\r\n";
const DEFAULT_PORT: u16 = 8080;

fn content_type(encoding: ByteStreamEncoding) -> Result<&'static str, G2gError> {
    match encoding {
        ByteStreamEncoding::MpegTs => Ok("video/mp2t"),
        ByteStreamEncoding::IsoBmff => Ok("video/mp4"),
        ByteStreamEncoding::Matroska => Ok("video/x-matroska"),
        _ => Err(G2gError::CapsMismatch),
    }
}

fn accepted_caps() -> Vec<Caps> {
    [
        ByteStreamEncoding::MpegTs,
        ByteStreamEncoding::IsoBmff,
        ByteStreamEncoding::Matroska,
    ]
    .into_iter()
    .map(|encoding| Caps::ByteStream { encoding })
    .collect()
}

fn encoding_of(caps: &Caps) -> Result<ByteStreamEncoding, G2gError> {
    match caps {
        Caps::ByteStream { encoding } => content_type(*encoding).map(|_| *encoding),
        _ => Err(G2gError::CapsMismatch),
    }
}

/// Serves a muxed MPEG-TS, fragmented MP4 or live Matroska stream to every HTTP
/// client that GETs it. A client joining mid-stream starts at the next keyframe,
/// after the fMP4 init segment or the Matroska header.
///
/// ```text
/// ... ! mpegtsmux ! httpserversink host=0.0.0.0 port=8080
/// ```
#[derive(Debug)]
pub struct HttpServerSink {
    host: String,
    port: u16,
    listener: Option<StdTcpListener>,
    bound_port: Option<u16>,
    encoding: Option<ByteStreamEncoding>,
    joins: Option<JoinPoints>,
    arrivals: Option<mpsc::UnboundedReceiver<TcpStream>>,
    acceptor: Option<AbortOnDrop>,
    // answered with headers, waiting for a frame they can start decoding at
    joining: Vec<TcpStream>,
    clients: Vec<TcpStream>,
    bytes_written: u64,
}

impl Default for HttpServerSink {
    fn default() -> Self {
        Self::new(DEFAULT_HOST, DEFAULT_PORT)
    }
}

impl HttpServerSink {
    /// Listen on `host:port`; port 0 lets the OS pick, read back with
    /// [`Self::current_port`].
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
            listener: None,
            bound_port: None,
            encoding: None,
            joins: None,
            arrivals: None,
            acceptor: None,
            joining: Vec::new(),
            clients: Vec::new(),
            bytes_written: 0,
        }
    }

    /// Bind now, so a caller that asked for port 0 can tell clients where to
    /// connect before the pipeline runs.
    pub fn bind(&mut self) -> Result<u16, G2gError> {
        if let Some(port) = self.bound_port {
            return Ok(port);
        }
        let listener = bind(&self.host, self.port)?;
        let port = listener.local_addr().map_err(io_err)?.port();
        self.listener = Some(listener);
        self.bound_port = Some(port);
        Ok(port)
    }

    pub fn current_port(&self) -> Option<u16> {
        self.bound_port
    }

    /// Clients receiving the stream, not counting ones waiting for a keyframe.
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    fn start_acceptor(&mut self, encoding: ByteStreamEncoding) -> Result<(), G2gError> {
        let Some(listener) = self.listener.take() else {
            return Ok(());
        };
        let listener = TcpListener::from_std(listener).map_err(io_err)?;
        let (arrived, arrivals) = mpsc::unbounded_channel();
        self.arrivals = Some(arrivals);
        self.acceptor = Some(AbortOnDrop(tokio::spawn(accept_stream_requests(
            listener,
            content_type(encoding)?,
            arrived,
        ))));
        Ok(())
    }

    async fn deliver(&mut self, frame: &[u8], keyframe: bool) -> Result<(), G2gError> {
        if let Some(arrivals) = &mut self.arrivals {
            while let Ok(stream) = arrivals.try_recv() {
                self.joining.push(stream);
            }
        }
        let joins = self.joins.as_mut().ok_or(G2gError::NotConfigured)?;
        let prefix = joins.prefix_for(frame, keyframe)?;
        if let Some(prefix) = prefix.filter(|_| !self.joining.is_empty()) {
            for mut stream in core::mem::take(&mut self.joining) {
                if prefix.is_empty() || write_chunk(&mut stream, &prefix).await.is_ok() {
                    self.clients.push(stream);
                }
            }
        }
        // no per-client queue: a slow client slows the pipeline, as tcpserversink does
        let mut index = 0;
        while index < self.clients.len() {
            match write_chunk(&mut self.clients[index], frame).await {
                Ok(()) => index += 1,
                Err(_) => {
                    self.clients.swap_remove(index);
                }
            }
        }
        self.bytes_written += frame.len() as u64;
        Ok(())
    }
}

// where a client joining a running stream can start, per container
#[derive(Debug)]
enum JoinPoints {
    MpegTs,
    Fmp4 {
        // the `ftyp` + `moov` a late client needs before any fragment
        init_segment: Vec<u8>,
        // the first track in the `moov`, whose sync fragments are the join points
        lead_track: Option<u32>,
    },
    Matroska(crate::matroska::LiveJoin),
}

impl JoinPoints {
    fn new(encoding: ByteStreamEncoding) -> Result<Self, G2gError> {
        match encoding {
            ByteStreamEncoding::MpegTs => Ok(JoinPoints::MpegTs),
            ByteStreamEncoding::IsoBmff => Ok(JoinPoints::Fmp4 {
                init_segment: Vec::new(),
                lead_track: None,
            }),
            ByteStreamEncoding::Matroska => Ok(JoinPoints::Matroska(Default::default())),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    // the bytes a waiting client needs ahead of `frame`, `None` when it cannot start there
    fn prefix_for(&mut self, frame: &[u8], keyframe: bool) -> Result<Option<Vec<u8>>, G2gError> {
        match self {
            JoinPoints::MpegTs => Ok(keyframe.then(Vec::new)),
            JoinPoints::Fmp4 {
                init_segment,
                lead_track,
            } => fmp4_prefix_for(init_segment, lead_track, frame),
            JoinPoints::Matroska(join) => join.prefix_for(frame),
        }
    }
}

fn fmp4_prefix_for(
    init_segment: &mut Vec<u8>,
    lead_track: &mut Option<u32>,
    frame: &[u8],
) -> Result<Option<Vec<u8>>, G2gError> {
    let mut carries_init = false;
    let mut lead_sync_fragment = false;
    let mut offset = 0usize;
    for (kind, payload) in boxes(frame) {
        let whole = &frame[offset..offset + payload.len() + 8];
        offset += whole.len();
        match kind {
            b"ftyp" | b"moov" => {
                if !carries_init {
                    init_segment.clear();
                }
                carries_init = true;
                if kind == b"moov" {
                    *lead_track = track_timescales(payload)?.first().map(|(id, _)| *id);
                }
                init_segment.extend_from_slice(whole);
            }
            b"moof" => {
                let (track, sync, _) = fragment_info(payload, frame.len())?;
                lead_sync_fragment |= sync && Some(track) == *lead_track;
            }
            _ => {}
        }
    }
    if carries_init {
        return Ok(Some(Vec::new()));
    }
    Ok(lead_sync_fragment.then(|| init_segment.clone()))
}

async fn write_chunk(stream: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    stream
        .write_all(alloc::format!("{:x}\r\n", bytes.len()).as_bytes())
        .await?;
    stream.write_all(bytes).await?;
    stream.write_all(b"\r\n").await
}

async fn accept_stream_requests(
    listener: TcpListener,
    content_type: &'static str,
    arrived: mpsc::UnboundedSender<TcpStream>,
) {
    while let Ok((stream, _peer)) = listener.accept().await {
        tokio::spawn(answer_stream_request(stream, content_type, arrived.clone()));
    }
}

async fn answer_stream_request(
    mut stream: TcpStream,
    content_type: &'static str,
    arrived: mpsc::UnboundedSender<TcpStream>,
) {
    let (head, streams) = match read_request(&mut stream).await {
        Err(_) => (
            response_head(Status::BadRequest, None, BodyLength::Exact(0)),
            false,
        ),
        Ok(request) => match request.method.as_str() {
            "GET" => (
                response_head(Status::Ok, Some(content_type), BodyLength::Chunked),
                true,
            ),
            "HEAD" => (
                response_head(Status::Ok, Some(content_type), BodyLength::Chunked),
                false,
            ),
            _ => (
                response_head(Status::MethodNotAllowed, None, BodyLength::Exact(0)),
                false,
            ),
        },
    };
    if stream.write_all(&head).await.is_err() || !streams {
        return;
    }
    // the receiver is gone once the run has ended
    let _ = arrived.send(stream);
}

impl AsyncElement for HttpServerSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        encoding_of(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::from_alternatives(accepted_caps()))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let encoding = encoding_of(absolute_caps)?;
        self.encoding = Some(encoding);
        self.joins = Some(JoinPoints::new(encoding)?);
        self.bind()?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let encoding = self.encoding.ok_or(G2gError::NotConfigured)?;
            self.start_acceptor(encoding)?;
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let bytes = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    self.deliver(bytes, frame.timing.keyframe).await?;
                }
                PipelinePacket::Eos => {
                    let mut streams = core::mem::take(&mut self.clients);
                    streams.append(&mut self.joining);
                    for mut stream in streams {
                        // a client that already left needs no ending
                        let _ = stream.write_all(LAST_CHUNK).await;
                    }
                }
                // the response headers already named the content type
                PipelinePacket::CapsChanged(caps) if encoding_of(&caps)? != encoding => {
                    return Err(G2gError::CapsMismatch);
                }
                _ => {}
            }
            Ok(())
        })
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "HTTP server sink",
            "Sink/Network",
            "Serves a muxed MPEG-TS, fragmented MP4 or live Matroska stream to HTTP clients",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        HTTPSERVERSINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "host" => self.host = value.as_str().ok_or(PropError::Type)?.into(),
            "port" => {
                let port = value.as_uint().ok_or(PropError::Type)?;
                self.port = u16::try_from(port).map_err(|_| PropError::Value)?;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "host" => Some(PropValue::Str(self.host.clone())),
            "port" => Some(PropValue::Uint(u64::from(self.port))),
            _ => None,
        }
    }
}

impl PadTemplates for HttpServerSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::from_alternatives(
            accepted_caps(),
        ))])
    }
}

static HTTPSERVERSINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new("host", PropKind::Str, "address to listen on").with_default(DEFAULT_HOST),
    PropertySpec::new(
        "port",
        PropKind::Uint,
        "port to listen on (0 = any free port)",
    )
    .with_default("8080")
    .with_range("0", "65535"),
];
