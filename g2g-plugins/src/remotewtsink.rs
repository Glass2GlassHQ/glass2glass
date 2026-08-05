//! Remote WebTransport sink (M901, `webtransport` feature): the WebTransport send
//! half of the distributed-graph primitive, the sibling of the TCP
//! [`RemoteSink`](crate::remotesink) and the WebSocket
//! [`RemoteWsSink`](crate::remotewssink).
//!
//! `RemoteWtSink` accepts *any* caps and forwards the whole `PipelinePacket`
//! stream (the leading `CapsChanged`, `Segment`, every `DataFrame`, mid-stream
//! caps refinement, `Flush`, `Eos`) over one WebTransport session, each packet
//! serialized by [`g2g_core::wire`]. The session's single bidirectional stream is
//! a QUIC stream, so it is an ordered reliable *byte* stream: the framing is the
//! TCP pair's `u32` length prefix, not the WebSocket pair's one-message-per-packet.
//! The receiving half is [`RemoteWtSrc`](crate::remotewtsrc).
//!
//! What WebTransport adds over the WebSocket carrier is the QUIC connection under
//! it: head-of-line blocking is per stream rather than per connection, the
//! handshake is 1-RTT, and a browser peer can speak it directly (`new
//! WebTransport(url)`) without a TLS-terminating proxy in front. Only CPU-memory
//! frames cross the wire; a device-resident frame yields
//! [`G2gError::UnsupportedDomain`](g2g_core::G2gError), exactly as any CPU sink
//! already requires.
//!
//! `RemoteWtSink` is the client (it dials the [`RemoteWtSrc`] server). The QUIC +
//! HTTP/3 CONNECT handshake is async, so as with the WebSocket sink the connect is
//! deferred to the first `process` call, where a runtime context is guaranteed.
//! QUIC is always TLS: the server's certificate must chain to a system root, or be
//! named by its SHA-256 digest in `server-certificate-hashes` (what a browser's
//! `serverCertificateHashes` option does for a self-signed certificate).
//!
//! The shared client machinery (caps-dedup, the reconnect/retry `deliver` loop,
//! the `AsyncElement` glue) lives in [`RemoteClient`](crate::remoteclient); this
//! file supplies only the WebTransport transport (`WtClient`).

use alloc::boxed::Box;
use alloc::string::{String, ToString};

use g2g_core::{
    AsyncElement, G2gError, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::remoteclient::{PacketClient, RemoteClient};
use crate::remotesource::TransportFuture;
use crate::remotewtio::{self, WtStream};

/// WebTransport `RemoteWtSink`: a length-framed [`g2g_core::wire`] stream over one
/// bidirectional WebTransport stream, received by
/// [`RemoteWtSrc`](crate::remotewtsrc).
pub type RemoteWtSink = RemoteClient<WtClient>;

impl RemoteWtSink {
    /// Send the packet stream to `url` (the [`RemoteWtSrc`](crate::remotewtsrc)
    /// server, e.g. `https://127.0.0.1:9603`).
    pub fn new(url: impl Into<String>) -> Self {
        RemoteClient::from_transport(WtClient::new(url))
    }

    /// Accept only server certificates whose SHA-256 digest is listed (hex,
    /// comma-separated), instead of requiring a system root. The same knob as the
    /// `server-certificate-hashes` property, set through the same path so a
    /// builder and a launch line cannot drift.
    pub fn with_server_certificate_hashes(mut self, hashes: impl Into<String>) -> Self {
        AsyncElement::set_property(
            &mut self,
            "server-certificate-hashes",
            PropValue::Str(hashes.into()),
        )
        .expect("server-certificate-hashes is a string");
        self
    }
}

/// WebTransport transport for [`RemoteClient`] (and, read back, for the
/// [`RemoteWtTransform`](crate::remotewttransform) round trip).
#[derive(Debug)]
pub struct WtClient {
    /// WebTransport URL of the server (e.g. `https://127.0.0.1:9603`).
    url: String,
    /// Comma-separated hex SHA-256 digests of the accepted server certificates;
    /// empty means "anything a system root signs".
    cert_hashes: String,
    /// Opened lazily on the first send (the QUIC / CONNECT handshake is async).
    stream: Option<WtStream>,
}

impl WtClient {
    /// Dial `url` on first use.
    pub(crate) fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            cert_hashes: String::new(),
            stream: None,
        }
    }

    /// The live session stream, for the transform's reply read.
    pub(crate) fn stream_mut(&mut self) -> Option<&mut WtStream> {
        self.stream.as_mut()
    }

    /// Shared by the sink and the transform: the two knobs a client has.
    pub(crate) fn set_client_prop(
        &mut self,
        name: &str,
        value: &PropValue,
    ) -> Option<Result<(), PropError>> {
        let target = match name {
            "location" => &mut self.url,
            "server-certificate-hashes" => &mut self.cert_hashes,
            _ => return None,
        };
        Some(match value.as_str() {
            Some(s) => {
                *target = s.to_string();
                Ok(())
            }
            None => Err(PropError::Type),
        })
    }

    pub(crate) fn get_client_prop(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.url.clone())),
            "server-certificate-hashes" => Some(PropValue::Str(self.cert_hashes.clone())),
            _ => None,
        }
    }
}

/// The `server-certificate-hashes` spec, shared by the sink and the transform.
pub(crate) const CERT_HASHES_PROP: PropertySpec = PropertySpec::new(
    "server-certificate-hashes",
    PropKind::Str,
    "accept only server certificates with these SHA-256 digests (hex, comma-separated); empty = system roots",
);

impl PacketClient for WtClient {
    const NAME: &'static str = "Remote WebTransport sink";
    const DESCRIPTION: &'static str =
        "Serializes the PipelinePacket stream and sends it over WebTransport to a remote RemoteWtSrc";
    const PROPERTIES: &'static [PropertySpec] = &[
        PropertySpec::new(
            "location",
            PropKind::Str,
            "WebTransport URL of the RemoteWtSrc server (e.g. https://host:port)",
        )
        .with_default("https://127.0.0.1:9603"),
        CERT_HASHES_PROP,
        PropertySpec::new(
            "reconnect-attempts",
            PropKind::Uint,
            "retry a failed connect / send up to N times (0 = off)",
        )
        .with_default("0"),
    ];

    fn is_connected(&self) -> bool {
        self.stream.is_some()
    }

    fn connect(&mut self) -> TransportFuture<'_, ()> {
        Box::pin(async move {
            self.stream = Some(remotewtio::connect(&self.url, &self.cert_hashes).await?);
            Ok(())
        })
    }

    fn send<'a>(&'a mut self, packet: &'a PipelinePacket) -> TransportFuture<'a, ()> {
        Box::pin(async move {
            let stream = self.stream.as_mut().ok_or(G2gError::NotConfigured)?;
            stream.send(packet).await
        })
    }

    fn reset(&mut self) {
        self.stream = None;
    }

    fn close(&mut self) -> TransportFuture<'_, ()> {
        Box::pin(async move {
            if let Some(stream) = self.stream.as_mut() {
                stream.finish().await;
            }
            Ok(())
        })
    }

    fn configure_connect(&mut self, _eager: bool) -> Result<(), G2gError> {
        // The QUIC + CONNECT handshake is async and needs a runtime, so the
        // connect is always deferred to the first send (as for the WebSocket sink).
        Ok(())
    }

    fn set_transport_prop(
        &mut self,
        name: &str,
        value: &PropValue,
    ) -> Option<Result<(), PropError>> {
        self.set_client_prop(name, value)
    }

    fn get_transport_prop(&self, name: &str) -> Option<PropValue> {
        self.get_client_prop(name)
    }
}
