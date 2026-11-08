use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use std::io;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_rustls::rustls::pki_types::ServerName;
use tokio_rustls::TlsConnector;
use url::{Host, Url};

use g2g_core::bus::{BusHandle, BusMessage};
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    g2g_error, g2g_warn, AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint,
    CapsSet, ConfigureOutcome, ElementMetadata, G2gError, HardwareError, NetworkFailure,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::ogg::{header_pages_len, OggCodec, OggDemuxer};
use crate::rtspserver::RtspMessage;
use crate::textencoding::base64_encode;
use crate::typefind::elementary_audio_caps;

const PLAIN_SCHEME: &str = "ice+http";
const TLS_SCHEME: &str = "ice+https";
const DEFAULT_PORT: u16 = 8000;
const DEFAULT_TIMEOUT_MS: u64 = 10_000;
const MAX_TIMEOUT_MS: u64 = 60_000;
const USER_AGENT: &str = concat!("g2g/", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamFormat {
    MpegAudio,
    Aac,
    Flac,
    Ogg,
}

impl StreamFormat {
    fn of(caps: &Caps) -> Option<Self> {
        match caps {
            Caps::Audio {
                format: AudioFormat::Mp2 | AudioFormat::Mp3,
                ..
            } => Some(Self::MpegAudio),
            Caps::Audio {
                format: AudioFormat::Aac,
                ..
            } => Some(Self::Aac),
            Caps::Audio {
                format: AudioFormat::Flac,
                ..
            } => Some(Self::Flac),
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Ogg,
            } => Some(Self::Ogg),
            _ => None,
        }
    }

    // Rocket Streaming Audio Server 1.3 refuses audio/ogg with a codecs parameter
    fn content_type(self) -> &'static str {
        match self {
            Self::MpegAudio => "audio/mpeg",
            Self::Aac => "audio/aac",
            Self::Flac => "audio/flac",
            Self::Ogg => "audio/ogg",
        }
    }
}

fn input_caps() -> CapsSet {
    let mut alternatives = [
        AudioFormat::Mp2,
        AudioFormat::Mp3,
        AudioFormat::Aac,
        AudioFormat::Flac,
    ]
    .map(elementary_audio_caps)
    .to_vec();
    alternatives.push(Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    });
    CapsSet::from_alternatives(alternatives)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AudioInfo {
    channels: u8,
    sample_rate: u32,
}

trait IcecastStream: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> IcecastStream for T {}

async fn within<T>(
    timeout: Duration,
    future: impl Future<Output = io::Result<T>>,
) -> io::Result<T> {
    tokio::time::timeout(timeout, future)
        .await
        .unwrap_or_else(|_| Err(io::ErrorKind::TimedOut.into()))
}

// gst reads at most this much of a server's answer
const MAX_RESPONSE_HEAD_BYTES: usize = 2048;
const READ_CHUNK_BYTES: usize = 512;

async fn read_response_head(stream: &mut dyn IcecastStream) -> io::Result<Vec<u8>> {
    let mut head = Vec::new();
    let mut chunk = [0u8; READ_CHUNK_BYTES];
    while RtspMessage::parse(&head).is_none() {
        if head.len() > MAX_RESPONSE_HEAD_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "response head too long",
            ));
        }
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        head.extend_from_slice(&chunk[..read]);
    }
    Ok(head)
}

fn status_code(start_line: &str) -> Option<u16> {
    let rest = start_line.strip_prefix("HTTP/1.")?;
    rest.split(' ').nth(1)?.parse().ok()
}

const STATUS_CONTINUE: u16 = 100;
const STATUS_SUCCESS: core::ops::RangeInclusive<u16> = 200..=204;

pub struct IcecastSink {
    location: Option<Url>,
    timeout_ms: u64,
    public: bool,
    stream_name: Option<String>,
    auto_reconnect: bool,
    format: Option<StreamFormat>,
    audio_info: Option<AudioInfo>,
    stream_headers: Vec<u8>,
    ogg_headers_found: bool,
    pending_ogg: Vec<u8>,
    connection: Option<Box<dyn IcecastStream>>,
    resend_headers: bool,
    bus: Option<BusHandle>,
    log_name: LogName,
}

impl core::fmt::Debug for IcecastSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IcecastSink")
            .field("location", &self.location.as_ref().map(Url::as_str))
            .field("format", &self.format)
            .field("connected", &self.connection.is_some())
            .finish_non_exhaustive()
    }
}

impl Default for IcecastSink {
    fn default() -> Self {
        Self {
            location: None,
            timeout_ms: DEFAULT_TIMEOUT_MS,
            public: true,
            stream_name: None,
            auto_reconnect: true,
            format: None,
            audio_info: None,
            stream_headers: Vec::new(),
            ogg_headers_found: false,
            pending_ogg: Vec::new(),
            connection: None,
            resend_headers: false,
            bus: None,
            log_name: LogName::default(),
        }
    }
}

impl LogSource for IcecastSink {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

fn parse_location(location: &str) -> Result<Url, PropError> {
    let url = Url::parse(location).map_err(|_| PropError::Value)?;
    if !matches!(url.scheme(), PLAIN_SCHEME | TLS_SCHEME) || url.host().is_none() {
        return Err(PropError::Value);
    }
    Ok(url)
}

impl IcecastSink {
    pub fn new(location: &str) -> Result<Self, PropError> {
        Ok(Self {
            location: Some(parse_location(location)?),
            ..Self::default()
        })
    }

    fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms)
    }

    fn put_request(&self, location: &Url, authority: &str, format: StreamFormat) -> String {
        let mut request = format!("PUT {} HTTP/1.1\r\nHost: {authority}\r\n", location.path());
        let password = location.password().unwrap_or("");
        if !location.username().is_empty() || !password.is_empty() {
            let credentials = format!("{}:{password}", location.username());
            request += &format!(
                "Authorization: Basic {}\r\n",
                base64_encode(credentials.as_bytes())
            );
        }
        request += &format!(
            "User-Agent: {USER_AGENT}\r\nContent-Type: {}\r\n",
            format.content_type()
        );
        if let Some(info) = self.audio_info {
            request += &format!(
                "Ice-audio-info: channels={};samplerate={}\r\n",
                info.channels, info.sample_rate
            );
        }
        request += &format!("Ice-public: {}\r\n", i32::from(self.public));
        if let Some(name) = &self.stream_name {
            request += &format!("Ice-name: {name}\r\n");
        }
        request + "Expect: 100-continue\r\n\r\n"
    }

    async fn connect(&self) -> Result<Box<dyn IcecastStream>, G2gError> {
        let location = self.location.as_ref().ok_or(G2gError::NotConfigured)?;
        let format = self.format.ok_or(G2gError::NotConfigured)?;
        let host = location.host().ok_or(G2gError::NotConfigured)?;
        let authority = format!("{host}:{}", location.port().unwrap_or(DEFAULT_PORT));
        let connect_failed = |error: io::Error| {
            g2g_error!(self, "cannot connect to {authority}: {error}");
            G2gError::Network(NetworkFailure::Connect)
        };
        let tcp = within(self.timeout(), TcpStream::connect(&authority))
            .await
            .map_err(connect_failed)?;
        let mut stream: Box<dyn IcecastStream> = if location.scheme() == TLS_SCHEME {
            let server_name = match host {
                Host::Domain(domain) => domain.to_string(),
                Host::Ipv4(address) => address.to_string(),
                Host::Ipv6(address) => address.to_string(),
            };
            let server_name = ServerName::try_from(server_name)
                .map_err(|error| connect_failed(io::Error::other(error)))?;
            let connector = TlsConnector::from(crate::tlsclient::client_config());
            Box::new(
                within(self.timeout(), connector.connect(server_name, tcp))
                    .await
                    .map_err(connect_failed)?,
            )
        } else {
            Box::new(tcp)
        };
        let request = self.put_request(location, &authority, format);
        within(self.timeout(), stream.write_all(request.as_bytes()))
            .await
            .map_err(connect_failed)?;
        let head = within(self.timeout(), read_response_head(stream.as_mut()))
            .await
            .map_err(connect_failed)?;
        let start_line = RtspMessage::parse(&head)
            .map(|message| message.start_line)
            .unwrap_or_default();
        let status = status_code(start_line).ok_or_else(|| {
            g2g_error!(self, "{authority} answered PUT with {start_line:?}");
            G2gError::Network(NetworkFailure::Connect)
        })?;
        if status != STATUS_CONTINUE && !STATUS_SUCCESS.contains(&status) {
            g2g_error!(
                self,
                "{authority} refused the source on mount {}: {start_line}",
                location.path()
            );
            return Err(G2gError::Hardware(HardwareError::Http(status)));
        }
        Ok(stream)
    }

    async fn send(&mut self, data: &[u8]) -> Result<(), G2gError> {
        let timeout = self.timeout();
        if self.connection.is_none() {
            let mut connection = self.connect().await?;
            if self.resend_headers {
                within(timeout, connection.write_all(&self.stream_headers))
                    .await
                    .map_err(|error| {
                        g2g_error!(self, "cannot resend the stream headers: {error}");
                        G2gError::Network(NetworkFailure::Stream)
                    })?;
                self.resend_headers = false;
            }
            self.connection = Some(connection);
        }
        let connection = self.connection.as_mut().ok_or(G2gError::NotConfigured)?;
        let Err(error) = within(timeout, connection.write_all(data)).await else {
            return Ok(());
        };
        if !self.auto_reconnect {
            g2g_error!(self, "cannot send to the server: {error}");
            return Err(G2gError::Network(NetworkFailure::Stream));
        }
        // gst drops the buffer that failed and reconnects on the next one
        g2g_warn!(self, "sending failed, reconnecting: {error}");
        if let Some(bus) = &self.bus {
            bus.try_post(BusMessage::Warning(G2gError::Network(
                NetworkFailure::Stream,
            )));
        }
        self.connection = None;
        self.resend_headers = true;
        Ok(())
    }

    async fn send_ogg(&mut self, data: &[u8]) -> Result<(), G2gError> {
        if self.ogg_headers_found {
            return self.send(data).await;
        }
        self.pending_ogg.extend_from_slice(data);
        let Some(headers_len) = header_pages_len(&self.pending_ogg) else {
            return Ok(());
        };
        let mut demuxer = OggDemuxer::new();
        demuxer.push_data(&self.pending_ogg[..headers_len]);
        let info = demuxer
            .info()
            .filter(|info| {
                matches!(
                    info.codec,
                    OggCodec::Opus | OggCodec::Vorbis | OggCodec::Flac
                )
            })
            .ok_or_else(|| {
                g2g_error!(
                    self,
                    "the Ogg stream opens with no Opus, Vorbis or FLAC stream"
                );
                G2gError::CapsMismatch
            })?;
        self.audio_info = Some(AudioInfo {
            channels: info.channels,
            sample_rate: info.sample_rate,
        });
        self.stream_headers = self.pending_ogg[..headers_len].to_vec();
        self.ogg_headers_found = true;
        let pending = core::mem::take(&mut self.pending_ogg);
        self.send(&pending).await
    }
}

impl AsyncElement for IcecastSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }

    fn set_bus(&mut self, bus: BusHandle) {
        self.bus = Some(bus);
    }

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(input_caps())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.location.is_none() {
            g2g_error!(self, "no location set");
            return Err(G2gError::NotConfigured);
        }
        let format = StreamFormat::of(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        if let Caps::Audio {
            channels,
            sample_rate,
            ..
        } = *absolute_caps
        {
            self.audio_info = (channels != 0 && sample_rate != 0).then_some(AudioInfo {
                channels,
                sample_rate,
            });
        }
        self.format = Some(format);
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
                        .require_system_slice(short_type_name::<Self>())?;
                    match self.format.ok_or(G2gError::NotConfigured)? {
                        StreamFormat::Ogg => return self.send_ogg(bytes).await,
                        StreamFormat::Flac if bytes.starts_with(b"fLaC") => {
                            self.stream_headers = bytes.to_vec();
                        }
                        _ => {}
                    }
                    self.send(bytes).await?;
                }
                PipelinePacket::Eos => {
                    if let Some(mut connection) = self.connection.take() {
                        within(self.timeout(), connection.shutdown())
                            .await
                            .map_err(|error| {
                                g2g_error!(self, "cannot close the connection: {error}");
                                G2gError::Network(NetworkFailure::Stream)
                            })?;
                    }
                }
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        ICECASTSINK_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Icecast Sink",
            "Sink/Network",
            "Sends an audio stream to an Icecast server",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.location = Some(parse_location(value.as_str().ok_or(PropError::Type)?)?);
            }
            "timeout" => {
                let timeout_ms = value.as_uint().ok_or(PropError::Type)?;
                if timeout_ms > MAX_TIMEOUT_MS {
                    return Err(PropError::Value);
                }
                self.timeout_ms = timeout_ms;
            }
            "public" => self.public = value.as_bool().ok_or(PropError::Type)?,
            "stream-name" => {
                self.stream_name = Some(value.as_str().ok_or(PropError::Type)?.to_string());
            }
            "auto-reconnect" => self.auto_reconnect = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(
                self.location
                    .as_ref()
                    .map(Url::to_string)
                    .unwrap_or_default(),
            )),
            "timeout" => Some(PropValue::Uint(self.timeout_ms)),
            "public" => Some(PropValue::Bool(self.public)),
            "stream-name" => Some(PropValue::Str(self.stream_name.clone().unwrap_or_default())),
            "auto-reconnect" => Some(PropValue::Bool(self.auto_reconnect)),
            _ => None,
        }
    }
}

static ICECASTSINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "location",
        PropKind::Str,
        "Icecast server, credentials and mount path, e.g. ice+http://source:p4ssw0rd@ingest.example.com:8000/radio",
    ),
    PropertySpec::new(
        "timeout",
        PropKind::Uint,
        "timeout for network activity, in milliseconds",
    )
    .with_default("10000")
    .with_range("0", "60000"),
    PropertySpec::new(
        "public",
        PropKind::Bool,
        "whether the stream should be listed on the server's stream directory",
    )
    .with_default("true"),
    PropertySpec::new(
        "stream-name",
        PropKind::Str,
        "name of the stream, if not configured server-side for the mount point",
    ),
    PropertySpec::new(
        "auto-reconnect",
        PropKind::Bool,
        "reconnect when the connection to the server breaks",
    )
    .with_default("true"),
];

impl PadTemplates for IcecastSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(input_caps())])
    }
}
