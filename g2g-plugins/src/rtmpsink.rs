//! RTMP egress sink (RtmpSink, `rtmp` feature): connects out to an RTMP server
//! (`rtmp://host[:port]/app/streamkey`) and *publishes* an incoming FLV byte
//! stream (`Caps::ByteStream{Flv}`, as produced by `flvmux`). The inverse of
//! [`RtmpSrc`](crate::rtmpsrc): the [`RtmpPublisher`](crate::rtmp::RtmpPublisher)
//! sans-IO client does the protocol work (handshake, the `connect` /
//! `createStream` / `publish` command ladder, FLV-tags -> RTMP messages); this
//! element is the tokio TCP I/O around it.
//!
//! Scope: one connection / one stream, the simple handshake, H.264 + AAC, AMF0.
//! The connection is opened lazily on the first buffer (so `flvmux`'s header
//! tag has been produced) and the publish ladder is driven to completion before
//! any media is sent. The server's Window Acknowledgement Size is honored as
//! back-pressure: when the unacknowledged in-flight byte count reaches the
//! window, the sink blocks on the socket for the server's Acknowledgement before
//! feeding more media, so a slow server throttles the pipeline instead of the
//! socket buffer bloating unboundedly.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, HardwareError, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::filesink::io_err;
use crate::rtmp::RtmpPublisher;

/// TCP read buffer for draining the server's control messages.
const READ_BUF: usize = 65_536;

/// The default RTMP port, used when the URL omits one.
const DEFAULT_RTMP_PORT: u16 = 1935;

/// A parsed `rtmp://host[:port]/app/streamkey` target.
#[derive(Debug, Clone)]
struct RtmpTarget {
    host: String,
    port: u16,
    app: String,
    tc_url: String,
    stream_key: String,
}

/// Parse an RTMP URL into its connection + publish fields. `app` is the first
/// path segment, `stream_key` the rest, `tc_url` the `rtmp://authority/app` the
/// server expects echoed in the `connect` command.
fn parse_rtmp_url(url: &str) -> Result<RtmpTarget, G2gError> {
    let rest = url
        .strip_prefix("rtmp://")
        .ok_or(G2gError::Hardware(HardwareError::Other))?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i + 1..]),
        None => (rest, ""),
    };
    if authority.is_empty() {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().unwrap_or(DEFAULT_RTMP_PORT)),
        None => (authority.to_string(), DEFAULT_RTMP_PORT),
    };
    let mut segs = path.splitn(2, '/');
    let app = segs.next().unwrap_or("").to_string();
    let stream_key = segs.next().unwrap_or("").to_string();
    Ok(RtmpTarget {
        host,
        port,
        app: app.clone(),
        tc_url: format!("rtmp://{authority}/{app}"),
        stream_key,
    })
}

#[derive(Debug)]
pub struct RtmpSink {
    url: String,
    target: Option<RtmpTarget>,
    stream: Option<TcpStream>,
    publisher: Option<RtmpPublisher>,
    bytes_sent: u64,
    frames_sent: u64,
    eos_seen: bool,
    /// Use the digest (complex) handshake strict CDNs require (default on; the
    /// C2 auto-falls-back to an echo against a simple server).
    complex_handshake: bool,
}

impl RtmpSink {
    /// Publish to `url` (`rtmp://host[:port]/app/streamkey`). The URL is parsed
    /// in `configure_pipeline`; the socket is opened on the first buffer.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            target: None,
            stream: None,
            publisher: None,
            bytes_sent: 0,
            frames_sent: 0,
            eos_seen: false,
            complex_handshake: true,
        }
    }

    /// Choose the RTMP handshake: the digest (complex) handshake (default,
    /// required by strict CDNs), or the plain simple handshake.
    pub fn with_complex_handshake(mut self, complex: bool) -> Self {
        self.complex_handshake = complex;
        self
    }

    pub fn bytes_sent(&self) -> u64 {
        self.bytes_sent
    }

    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    pub fn eos_seen(&self) -> bool {
        self.eos_seen
    }

    fn input_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Flv,
        }
    }
}

/// Write the publisher's queued bytes (handshake / commands / media) to the socket.
async fn flush_out(
    stream: &mut TcpStream,
    publisher: &mut RtmpPublisher,
) -> Result<usize, G2gError> {
    let out = publisher.take_outbound();
    if out.is_empty() {
        return Ok(0);
    }
    stream.write_all(&out).await.map_err(io_err)?;
    Ok(out.len())
}

/// Drive the C0/C1 handshake and the connect/createStream/publish ladder to
/// completion, exchanging bytes with the server until media may flow.
async fn drive_publish(
    stream: &mut TcpStream,
    publisher: &mut RtmpPublisher,
) -> Result<(), G2gError> {
    flush_out(stream, publisher).await?; // C0 + C1
    let mut buf = [0u8; READ_BUF];
    while !publisher.is_publishing() {
        let n = stream.read(&mut buf).await.map_err(io_err)?;
        if n == 0 {
            return Err(G2gError::Hardware(HardwareError::Other)); // server closed before publish
        }
        publisher.push(&buf[..n]);
        flush_out(stream, publisher).await?;
    }
    Ok(())
}

/// Non-blocking drain of any server bytes (control messages: window-ack /
/// set-peer-bw / acknowledgement), so the socket's receive buffer does not back
/// up mid-stream and the publisher's ack window advances.
fn drain_incoming(stream: &TcpStream, publisher: &mut RtmpPublisher) {
    let mut buf = [0u8; READ_BUF];
    loop {
        match stream.try_read(&mut buf) {
            Ok(0) => break, // closed; surfaced on the next write
            Ok(n) => publisher.push(&buf[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(_) => break,
        }
    }
}

/// Honor the server's Window Acknowledgement Size (RFC §5.4.3): while the
/// publisher's unacknowledged in-flight bytes have hit the window, block on the
/// socket for the server's Acknowledgement (which advances the window). A bounded
/// timeout keeps a silent server from wedging the pipeline forever; on timeout we
/// proceed (best-effort, as most servers ack promptly).
async fn await_ack_window(
    stream: &mut TcpStream,
    publisher: &mut RtmpPublisher,
) -> Result<(), G2gError> {
    use core::time::Duration;
    let mut buf = [0u8; READ_BUF];
    while publisher.throttled() {
        match tokio::time::timeout(Duration::from_secs(5), stream.read(&mut buf)).await {
            Ok(Ok(0)) => return Err(G2gError::Hardware(HardwareError::Other)), // server closed
            Ok(Ok(n)) => publisher.push(&buf[..n]),
            Ok(Err(e)) => return Err(io_err(e)),
            Err(_) => break, // timed out waiting for an ack: proceed best-effort
        }
    }
    Ok(())
}

impl AsyncElement for RtmpSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(Self::input_caps()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.target = Some(parse_rtmp_url(&self.url)?);
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
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    // Lazily connect + publish on the first buffer.
                    if self.stream.is_none() {
                        let target = self.target.clone().ok_or(G2gError::NotConfigured)?;
                        let mut stream = TcpStream::connect((target.host.as_str(), target.port))
                            .await
                            .map_err(io_err)?;
                        let mut publisher =
                            RtmpPublisher::new(target.app, target.tc_url, target.stream_key)
                                .with_complex_handshake(self.complex_handshake);
                        drive_publish(&mut stream, &mut publisher).await?;
                        self.stream = Some(stream);
                        self.publisher = Some(publisher);
                    }
                    let stream = self.stream.as_mut().ok_or(G2gError::NotConfigured)?;
                    let publisher = self.publisher.as_mut().ok_or(G2gError::NotConfigured)?;
                    drain_incoming(stream, publisher);
                    // Apply the server's ack-window back-pressure before queuing
                    // more media, so the socket buffer does not bloat unboundedly.
                    await_ack_window(stream, publisher).await?;
                    publisher.push_flv(slice);
                    self.bytes_sent += flush_out(stream, publisher).await? as u64;
                    self.frames_sent += 1;
                }
                PipelinePacket::Eos => {
                    if let (Some(stream), Some(publisher)) =
                        (self.stream.as_mut(), self.publisher.as_mut())
                    {
                        flush_out(stream, publisher).await?;
                        let _ = stream.shutdown().await;
                    }
                    self.eos_seen = true;
                }
                // No publish-side equivalent: a flushing seek is meaningless on a
                // live egress stream, and caps/segment are control only.
                PipelinePacket::Flush
                | PipelinePacket::CapsChanged(_)
                | PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        RTMPSINK_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "RTMP egress sink",
            "Sink/Network",
            "Publishes an FLV byte stream to an RTMP server",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.url = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "complex-handshake" => {
                self.complex_handshake = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.url.clone())),
            "complex-handshake" => Some(PropValue::Bool(self.complex_handshake)),
            _ => None,
        }
    }
}

/// `RtmpSink`'s settable properties: the publish URL + the handshake mode.
static RTMPSINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "location",
        PropKind::Str,
        "rtmp://host[:port]/app/streamkey",
    ),
    PropertySpec::new(
        "complex-handshake",
        PropKind::Bool,
        "use the digest handshake strict CDNs require (default true)",
    ),
];

impl PadTemplates for RtmpSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::one(Self::input_caps()))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_rtmp_url() {
        let t = parse_rtmp_url("rtmp://example.com:1936/live/streamkey123").unwrap();
        assert_eq!(t.host, "example.com");
        assert_eq!(t.port, 1936);
        assert_eq!(t.app, "live");
        assert_eq!(t.stream_key, "streamkey123");
        assert_eq!(t.tc_url, "rtmp://example.com:1936/live");
    }

    #[test]
    fn defaults_port_and_splits_multi_segment_key() {
        let t = parse_rtmp_url("rtmp://host/app/path/to/key").unwrap();
        assert_eq!(t.port, DEFAULT_RTMP_PORT);
        assert_eq!(t.app, "app");
        assert_eq!(t.stream_key, "path/to/key", "the key keeps its slashes");
        assert_eq!(t.tc_url, "rtmp://host/app");
    }

    #[test]
    fn rejects_non_rtmp_scheme() {
        assert!(parse_rtmp_url("http://host/app").is_err());
        assert!(parse_rtmp_url("rtmp:///app").is_err());
    }
}
