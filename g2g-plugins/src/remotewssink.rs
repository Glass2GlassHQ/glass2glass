//! Remote WebSocket transport sink (M554, `remote-ws` feature): the WebSocket
//! send half of the distributed-graph primitive, the sibling of the TCP
//! [`RemoteSink`](crate::remotesink).
//!
//! `RemoteWsSink` accepts *any* caps and forwards the whole `PipelinePacket`
//! stream (the leading `CapsChanged`, `Segment`, every `DataFrame`, mid-stream
//! caps refinement, `Flush`, `Eos`) over a WebSocket connection, each packet
//! serialized by [`g2g_core::wire`] and sent as one binary WebSocket message.
//! WebSocket is already message-framed, so unlike the TCP sink there is no `u32`
//! length prefix: one `encode_packet` body == one `Message::Binary`. The
//! receiving half is [`RemoteWsSrc`](crate::remotewssrc).
//!
//! The point of the WebSocket variant is reach: a browser peer can speak only
//! WebSocket (not a raw TCP socket), so this is the transport that lets a
//! `g2g-web` graph ship an edge to a native subgraph over the same wire codec,
//! turning the bespoke M549 browser->detect-server shim into an instance of the
//! media-agnostic primitive.
//!
//! `RemoteWsSink` is the WebSocket *client* (it dials the [`RemoteWsSrc`]
//! server). The WebSocket handshake is async, so unlike the TCP sink (which
//! connects synchronously in `configure_pipeline`) the connect is deferred to
//! the first `process` call, where a runtime context is guaranteed; the caps are
//! sent as the first wire message (so the server discovers the media type from
//! the stream), then each subsequent packet follows. Only CPU-memory frames
//! cross the wire; a device-resident frame yields
//! [`G2gError::UnsupportedDomain`](g2g_core::G2gError), exactly as any CPU sink
//! already requires.
//!
//! The shared client machinery (caps-dedup, the reconnect/retry `deliver` loop,
//! the `AsyncElement` glue) lives in [`RemoteClient`](crate::remoteclient); this
//! file supplies only the WebSocket transport (`WsClient`).

use alloc::boxed::Box;
use alloc::string::{String, ToString};

use tokio::net::TcpStream;
use tokio_tungstenite::{accept_async, connect_async, MaybeTlsStream, WebSocketStream};

use g2g_core::{G2gError, PipelinePacket, PropError, PropKind, PropValue, PropertySpec};

use crate::filesink::io_err;
use crate::remoteclient::{PacketClient, RemoteClient};
use crate::remotesource::TransportFuture;
use crate::remotewsio::{bind_addr_of, send_wire, ws_err};

/// WebSocket `RemoteWsSink`: a [`g2g_core::wire`] stream carried one packet per
/// binary WebSocket message, received by [`RemoteWsSrc`](crate::remotewssrc).
pub type RemoteWsSink = RemoteClient<WsClient>;

impl RemoteWsSink {
    /// Send the packet stream to `url` (the [`RemoteWsSrc`](crate::remotewssrc)
    /// server, e.g. `ws://127.0.0.1:9601`).
    pub fn new(url: impl Into<String>) -> Self {
        RemoteClient::from_transport(WsClient::new(url))
    }
}

/// WebSocket transport for [`RemoteClient`] (and, read back, for the
/// [`RemoteWsTransform`](crate::remotewstransform) round trip).
#[derive(Debug)]
pub struct WsClient {
    /// WebSocket URL of the `RemoteWsSrc` server (e.g. `ws://127.0.0.1:9601`),
    /// or the address to bind when `listen` is set.
    url: String,
    /// Wait for a client to dial in rather than dialing out (the `listen`
    /// property): the receiving peer is then whoever connects, which is the only
    /// direction a browser can take.
    listen: bool,
    /// Opened lazily on the first send (the handshake is async).
    socket: Option<WebSocketStream<MaybeTlsStream<TcpStream>>>,
}

impl WsClient {
    /// Dial `url` on first use.
    pub(crate) fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            listen: false,
            socket: None,
        }
    }

    /// The live socket, for the transform's reply read.
    pub(crate) fn socket_mut(&mut self) -> Option<&mut WebSocketStream<MaybeTlsStream<TcpStream>>> {
        self.socket.as_mut()
    }
}

impl PacketClient for WsClient {
    const NAME: &'static str = "Remote WebSocket sink";
    const DESCRIPTION: &'static str =
        "Serializes the PipelinePacket stream and sends it over a WebSocket to a remote RemoteWsSrc";
    const PROPERTIES: &'static [PropertySpec] = &[
        PropertySpec::new(
            "location",
            PropKind::Str,
            "WebSocket URL of the RemoteWsSrc server to connect to (e.g. ws://host:port)",
        )
        .with_default("ws://127.0.0.1:9601"),
        PropertySpec::new(
            "reconnect-attempts",
            PropKind::Uint,
            "retry a failed connect / send up to N times (0 = off)",
        )
        .with_default("0"),
        PropertySpec::new(
            "listen",
            PropKind::Bool,
            "serve the stream to a client that dials in, instead of dialing out",
        )
        .with_default("false"),
    ];

    fn is_connected(&self) -> bool {
        self.socket.is_some()
    }

    fn connect(&mut self) -> TransportFuture<'_, ()> {
        Box::pin(async move {
            if self.listen {
                let bind = bind_addr_of(&self.url)?;
                let listener = tokio::net::TcpListener::bind(bind).await.map_err(io_err)?;
                let (tcp, _peer) = listener.accept().await.map_err(io_err)?;
                // The accepted stream wears the client side's type, so one field
                // holds either role's socket.
                let socket = accept_async(MaybeTlsStream::Plain(tcp))
                    .await
                    .map_err(ws_err)?;
                self.socket = Some(socket);
                return Ok(());
            }
            let (socket, _resp) = connect_async(&self.url).await.map_err(ws_err)?;
            self.socket = Some(socket);
            Ok(())
        })
    }

    fn send<'a>(&'a mut self, packet: &'a PipelinePacket) -> TransportFuture<'a, ()> {
        Box::pin(async move {
            let sock = self.socket.as_mut().ok_or(G2gError::NotConfigured)?;
            send_wire(sock, packet).await
        })
    }

    fn reset(&mut self) {
        self.socket = None;
    }

    fn close(&mut self) -> TransportFuture<'_, ()> {
        Box::pin(async move {
            if let Some(sock) = self.socket.as_mut() {
                let _ = sock.close(None).await;
            }
            Ok(())
        })
    }

    fn configure_connect(&mut self, _eager: bool) -> Result<(), G2gError> {
        // The WebSocket handshake is async and needs a runtime, so the connect is
        // always deferred to the first send (unlike the TCP sink's eager connect).
        Ok(())
    }

    fn set_transport_prop(
        &mut self,
        name: &str,
        value: &PropValue,
    ) -> Option<Result<(), PropError>> {
        match name {
            "location" => Some(match value.as_str() {
                Some(s) => {
                    self.url = s.to_string();
                    Ok(())
                }
                None => Err(PropError::Type),
            }),
            "listen" => Some(match value.as_bool() {
                Some(b) => {
                    self.listen = b;
                    Ok(())
                }
                None => Err(PropError::Type),
            }),
            _ => None,
        }
    }

    fn get_transport_prop(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.url.clone())),
            "listen" => Some(PropValue::Bool(self.listen)),
            _ => None,
        }
    }
}
