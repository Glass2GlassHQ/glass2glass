//! Remote WebSocket transport source (M554, `remote-ws` feature): the WebSocket
//! receive half of the distributed-graph primitive, the inverse of
//! [`RemoteWsSink`](crate::remotewssink) and the sibling of the TCP
//! [`RemoteSrc`](crate::remotesrc).
//!
//! `RemoteWsSrc` is the WebSocket *server*: it listens on a TCP port, accepts one
//! client connection, performs the WebSocket handshake, and reconstructs the
//! `PipelinePacket` stream the sender serialized ([`g2g_core::wire`], one packet
//! per binary WebSocket message). It is media-agnostic: the stream's first binary
//! message is the sender's negotiated `CapsChanged`, so the source *discovers*
//! the media type from the wire in `intercept_caps` (the async caps-discovery
//! pattern `RtspSrc` / `RemoteSrc` use), then re-emits the leading `CapsChanged`
//! and every subsequent packet in `run`, ending on the sender's `Eos` (or a clean
//! WebSocket close).
//!
//! The client may be a native [`RemoteWsSink`] or a browser `g2g-web` element
//! (`WsWireSink`) speaking the identical wire codec: the server does not care
//! which, so a decoded-frame edge cut in a browser graph lands here and the
//! downstream native subgraph runs exactly as it would locally.
//!
//! The shared server machinery lives in [`RemoteSource`](crate::remotesource);
//! this file supplies only the WebSocket transport (`WsTransport`).

use alloc::boxed::Box;

use std::net::{SocketAddr, TcpListener as StdTcpListener};

use tokio::net::TcpStream;
use tokio_tungstenite::{accept_async, WebSocketStream};

use g2g_core::{Caps, PipelinePacket, PropKind, PropertySpec};

use crate::filesink::io_err;
use crate::remotesource::{
    leading_caps, listen_tcp, PacketTransport, RemoteSource, TransportFuture,
};
use crate::remotewsio::recv_wire;

/// WebSocket `RemoteWsSrc`: a [`g2g_core::wire`] stream carried one packet per
/// binary WebSocket message, from a native `RemoteWsSink` or a browser
/// `WsWireSink`.
pub type RemoteWsSrc = RemoteSource<WsTransport>;

/// WebSocket transport for [`RemoteSource`].
#[derive(Debug, Default)]
pub struct WsTransport;

impl PacketTransport for WsTransport {
    type Conn = WebSocketStream<TcpStream>;
    type Listener = tokio::net::TcpListener;
    const NAME: &'static str = "Remote WebSocket source";
    const DESCRIPTION: &'static str =
        "Receives a serialized PipelinePacket stream over a WebSocket from a remote RemoteWsSink";
    const PROPERTIES: &'static [PropertySpec] = &[
        PropertySpec::new(
            "address",
            PropKind::Str,
            "local bind address (IP to listen on)",
        )
        .with_default("0.0.0.0"),
        PropertySpec::new(
            "port",
            PropKind::Uint,
            "local TCP port to listen on for WebSocket clients",
        )
        .with_range("0", "65535"),
        PropertySpec::new(
            "keep-listening",
            PropKind::Bool,
            "accept a replacement client when one drops without Eos",
        )
        .with_default("false"),
    ];

    fn listen(
        &mut self,
        bind: SocketAddr,
        adopt: Option<StdTcpListener>,
    ) -> TransportFuture<'_, Self::Listener> {
        Box::pin(async move { listen_tcp(bind, adopt).await })
    }

    fn listen_addr(listener: &Self::Listener) -> Option<SocketAddr> {
        listener.local_addr().ok()
    }

    fn accept(listener: &mut Self::Listener) -> TransportFuture<'_, (Self::Conn, Caps)> {
        Box::pin(async move {
            let (tcp, _peer) = listener.accept().await.map_err(io_err)?;
            let mut socket = accept_async(tcp).await.map_err(crate::remotewsio::ws_err)?;
            let caps = leading_caps(recv_wire(&mut socket).await?)?;
            Ok((socket, caps))
        })
    }

    fn recv(conn: &mut Self::Conn) -> TransportFuture<'_, Option<PipelinePacket>> {
        Box::pin(async move { recv_wire(conn).await })
    }
}
