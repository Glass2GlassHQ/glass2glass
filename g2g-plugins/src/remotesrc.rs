//! Remote transport source (M551, `remote` feature): the receive half of the
//! distributed-graph primitive, the inverse of [`RemoteSink`](crate::remotesink).
//!
//! `RemoteSrc` is the TCP *server*: it listens, accepts one [`RemoteSink`]
//! connection, and reconstructs the `PipelinePacket` stream the sink serialized
//! ([`g2g_core::wire`], length-framed). It is media-agnostic: the stream's first
//! wire packet is the sender's negotiated `CapsChanged`, so the source
//! *discovers* the media type from the wire in `intercept_caps` (the async
//! caps-discovery pattern `RtspSrc` uses), then re-emits the leading
//! `CapsChanged` and every subsequent packet (`Segment`, `DataFrame`s,
//! mid-stream caps refinement, `Flush`) in `run`, ending on the sender's `Eos`
//! (or a clean connection close). The downstream half of a split graph runs
//! exactly as it would locally; only the edge crossed a machine boundary.
//!
//! The shared server machinery lives in [`RemoteSource`](crate::remotesource);
//! this file supplies only the TCP transport (`TcpTransport`).

use alloc::boxed::Box;

use std::net::{SocketAddr, TcpListener as StdTcpListener};

use g2g_core::{Caps, PipelinePacket, PropKind, PropertySpec};

use crate::remotesource::{
    leading_caps, listen_tcp, PacketTransport, RemoteSource, TransportFuture,
};
use crate::remotewire::recv_framed;

/// TCP `RemoteSrc`: a length-framed [`g2g_core::wire`] stream over a plain TCP
/// connection, dialed by [`RemoteSink`](crate::remotesink).
pub type RemoteSrc = RemoteSource<TcpTransport>;

/// TCP transport for [`RemoteSource`].
#[derive(Debug, Default)]
pub struct TcpTransport;

impl PacketTransport for TcpTransport {
    type Conn = tokio::net::TcpStream;
    type Listener = tokio::net::TcpListener;
    const NAME: &'static str = "Remote source";
    const DESCRIPTION: &'static str =
        "Receives a serialized PipelinePacket stream over TCP from a remote RemoteSink";
    const PROPERTIES: &'static [PropertySpec] = &[
        PropertySpec::new(
            "address",
            PropKind::Str,
            "local bind address (IP to listen on)",
        )
        .with_default("0.0.0.0"),
        PropertySpec::new("port", PropKind::Uint, "local TCP port to listen on")
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
            let (mut stream, _peer) = listener.accept().await.map_err(crate::filesink::io_err)?;
            let caps = leading_caps(recv_framed(&mut stream).await?)?;
            Ok((stream, caps))
        })
    }

    fn recv(conn: &mut Self::Conn) -> TransportFuture<'_, Option<PipelinePacket>> {
        Box::pin(async move { recv_framed(conn).await })
    }
}
