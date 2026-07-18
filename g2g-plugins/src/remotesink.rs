//! Remote transport sink (M551, `remote` feature): the send half of the
//! distributed-graph primitive.
//!
//! `RemoteSink` accepts *any* caps and forwards the whole `PipelinePacket`
//! stream (the leading `CapsChanged`, `Segment`, every `DataFrame`, mid-stream
//! caps refinement, `Flush`, `Eos`) over a TCP connection, each packet
//! serialized by [`g2g_core::wire`] and length-framed (`u32` LE length, then the
//! wire body). The receiving half is [`RemoteSrc`](crate::remotesrc), which
//! deserializes the identical stream and re-emits it downstream. Together they
//! let you cut any edge in a graph and run the downstream subgraph in another
//! process or on another machine, without rewriting the graph: replace the edge
//! with `... ! remotesink` on the near side and `remotesrc ! ...` on the far
//! side. This is the general, media-agnostic form of the bespoke M549
//! browser-to-server detect offload.
//!
//! `RemoteSink` is the TCP *client* (it dials the [`RemoteSrc`] server). It
//! connects in `configure_pipeline` and sends the negotiated caps as the first
//! wire packet (so the server discovers the media type from the stream), then
//! forwards each subsequent packet from `process`. Only CPU-memory frames cross
//! the wire; a device-resident frame yields
//! [`G2gError::UnsupportedDomain`](g2g_core::G2gError) (put a download element
//! before the sink), matching what any CPU sink already requires.
//!
//! The shared client machinery (caps-dedup, the reconnect/retry `deliver` loop,
//! the `AsyncElement` glue) lives in [`RemoteClient`](crate::remoteclient); this
//! file supplies only the TCP transport (`TcpClient`).
//!
//! # Reconnection
//!
//! By default a broken connection (or a downstream node that is not up yet) is a
//! hard error that ends the pipeline. [`with_reconnect`](RemoteClient::with_reconnect)
//! (or the `reconnect-attempts` property) makes the sink resilient: the initial
//! connect is deferred and retried with a short backoff, and a mid-stream send
//! failure drops the dead socket, reconnects, and re-sends the current caps
//! (which the far side needs as its first packet) before retrying the packet. So
//! a `RemoteSrc` that starts late or restarts (paired with its own
//! [`with_reconnect`](crate::remotesrc::RemoteSrc::with_reconnect) keep-listening)
//! is transparently tolerated, up to the attempt budget.

use alloc::boxed::Box;

use std::net::{SocketAddr, TcpStream as StdTcpStream};

use tokio::io::AsyncWriteExt;

use g2g_core::wire::encode_packet;
use g2g_core::{G2gError, PipelinePacket, PropError, PropKind, PropValue, PropertySpec};

use crate::filesink::io_err;
use crate::remoteclient::{PacketClient, RemoteClient};
use crate::remotesource::TransportFuture;
use crate::remotewire::map_wire;

/// TCP `RemoteSink`: a length-framed [`g2g_core::wire`] stream over a plain TCP
/// connection, received by [`RemoteSrc`](crate::remotesrc).
pub type RemoteSink = RemoteClient<TcpClient>;

impl RemoteSink {
    /// Send the packet stream to `dest` (the [`RemoteSrc`](crate::remotesrc)
    /// listener, e.g. `127.0.0.1:9600`).
    pub fn new(dest: SocketAddr) -> Self {
        RemoteClient::from_transport(TcpClient {
            dest,
            std_stream: None,
            socket: None,
        })
    }
}

/// TCP transport for [`RemoteClient`].
#[derive(Debug)]
pub struct TcpClient {
    dest: SocketAddr,
    /// Connected synchronously in `configure_pipeline` (no runtime needed) when
    /// reconnect is off; wrapped into the tokio stream lazily on first `connect`.
    std_stream: Option<StdTcpStream>,
    socket: Option<tokio::net::TcpStream>,
}

impl TcpClient {
    /// Length-frame and write one already-encoded wire body.
    async fn write_frame(sock: &mut tokio::net::TcpStream, body: &[u8]) -> Result<(), G2gError> {
        sock.write_all(&(body.len() as u32).to_le_bytes())
            .await
            .map_err(io_err)?;
        sock.write_all(body).await.map_err(io_err)?;
        Ok(())
    }
}

impl PacketClient for TcpClient {
    const NAME: &'static str = "Remote sink";
    const DESCRIPTION: &'static str =
        "Serializes the PipelinePacket stream and sends it over TCP to a remote RemoteSrc";
    const PROPERTIES: &'static [PropertySpec] = &[
        PropertySpec::new(
            "host",
            PropKind::Str,
            "remote host to connect to (RemoteSrc address)",
        )
        .with_default("127.0.0.1"),
        PropertySpec::new("port", PropKind::Uint, "remote TCP port to connect to")
            .with_range("0", "65535"),
        PropertySpec::new(
            "reconnect-attempts",
            PropKind::Uint,
            "retry a failed connect / send up to N times (0 = off)",
        )
        .with_default("0"),
    ];

    fn is_connected(&self) -> bool {
        self.socket.is_some()
    }

    fn connect(&mut self) -> TransportFuture<'_, ()> {
        Box::pin(async move {
            // Reuse the eagerly-connected std stream if present, else dial `dest`.
            let sock = match self.std_stream.take() {
                Some(std) => tokio::net::TcpStream::from_std(std).map_err(io_err)?,
                None => tokio::net::TcpStream::connect(self.dest)
                    .await
                    .map_err(io_err)?,
            };
            self.socket = Some(sock);
            Ok(())
        })
    }

    fn send<'a>(&'a mut self, packet: &'a PipelinePacket) -> TransportFuture<'a, ()> {
        Box::pin(async move {
            let body = encode_packet(packet).map_err(map_wire)?;
            let sock = self.socket.as_mut().ok_or(G2gError::NotConfigured)?;
            Self::write_frame(sock, &body).await
        })
    }

    fn reset(&mut self) {
        self.socket = None;
        self.std_stream = None;
    }

    fn close(&mut self) -> TransportFuture<'_, ()> {
        Box::pin(async move {
            if let Some(sock) = self.socket.as_mut() {
                let _ = sock.shutdown().await;
            }
            Ok(())
        })
    }

    fn configure_connect(&mut self, eager: bool) -> Result<(), G2gError> {
        if eager && self.std_stream.is_none() && self.socket.is_none() {
            let stream = StdTcpStream::connect(self.dest).map_err(io_err)?;
            stream.set_nonblocking(true).map_err(io_err)?;
            self.std_stream = Some(stream);
        }
        Ok(())
    }

    fn set_transport_prop(
        &mut self,
        name: &str,
        value: &PropValue,
    ) -> Option<Result<(), PropError>> {
        crate::netprop::set_addr_prop(&mut self.dest, "host", name, value)
    }

    fn get_transport_prop(&self, name: &str) -> Option<PropValue> {
        crate::netprop::get_addr_prop(&self.dest, "host", name)
    }
}
