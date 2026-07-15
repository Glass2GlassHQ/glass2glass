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
use alloc::vec;
use alloc::vec::Vec;

use tokio::io::AsyncReadExt;

use g2g_core::wire::decode_packet;
use g2g_core::{
    Caps, G2gError, HardwareError, PipelinePacket, PropKind, PropertySpec,
};

use crate::filesink::io_err;
use crate::remotesource::{PacketTransport, RemoteSource, TransportFuture};
use crate::remotewire::map_wire;

/// TCP `RemoteSrc`: a length-framed [`g2g_core::wire`] stream over a plain TCP
/// connection, dialed by [`RemoteSink`](crate::remotesink).
pub type RemoteSrc = RemoteSource<TcpTransport>;

/// TCP transport for [`RemoteSource`].
#[derive(Debug)]
pub struct TcpTransport;

impl PacketTransport for TcpTransport {
    type Conn = tokio::net::TcpStream;
    const NAME: &'static str = "Remote source";
    const DESCRIPTION: &'static str =
        "Receives a serialized PipelinePacket stream over TCP from a remote RemoteSink";
    const PROPERTIES: &'static [PropertySpec] = &[
        PropertySpec::new("address", PropKind::Str, "local bind address (IP to listen on)")
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

    fn accept(listener: &tokio::net::TcpListener) -> TransportFuture<'_, (Self::Conn, Caps)> {
        Box::pin(async move {
            let (mut stream, _peer) = listener.accept().await.map_err(io_err)?;
            let body = read_frame(&mut stream).await?.ok_or(G2gError::NotConfigured)?;
            let caps = match decode_packet(&body).map_err(map_wire)? {
                PipelinePacket::CapsChanged(caps) => caps,
                // Any other first packet violates the protocol.
                _ => return Err(G2gError::Hardware(HardwareError::Other)),
            };
            Ok((stream, caps))
        })
    }

    fn recv(conn: &mut Self::Conn) -> TransportFuture<'_, Option<PipelinePacket>> {
        Box::pin(async move {
            match read_frame(conn).await? {
                Some(body) => Ok(Some(decode_packet(&body).map_err(map_wire)?)),
                None => Ok(None),
            }
        })
    }
}

/// Read one length-framed wire message. `Ok(None)` on a clean connection close at
/// a frame boundary (the stream's natural end).
async fn read_frame(sock: &mut tokio::net::TcpStream) -> Result<Option<Vec<u8>>, G2gError> {
    let mut len = [0u8; 4];
    match sock.read_exact(&mut len).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(io_err(e)),
    }
    let n = u32::from_le_bytes(len) as usize;
    let mut body = vec![0u8; n];
    sock.read_exact(&mut body).await.map_err(io_err)?;
    Ok(Some(body))
}
