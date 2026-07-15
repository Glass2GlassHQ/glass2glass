//! Native WebSocket <-> wire-codec I/O shared by the `remote-ws` elements
//! (`RemoteWsSink`, `RemoteWsSrc`, `RemoteWsTransform`): send one
//! [`PipelinePacket`] as a single binary WebSocket message, and read the next
//! packet back, skipping WebSocket control / text frames. WebSocket is already
//! message-framed, so one [`encode_packet`] body is one `Message::Binary` (no
//! `u32` length prefix, unlike the TCP pair). Generic over the underlying stream
//! so the client (`MaybeTlsStream<TcpStream>`) and server (`TcpStream`) sides
//! share the exact same framing.

use alloc::vec::Vec;

use futures_util::{SinkExt, StreamExt};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio_tungstenite::tungstenite::{Error as WsError, Message};
use tokio_tungstenite::WebSocketStream;

use g2g_core::wire::{decode_packet, encode_packet};
use g2g_core::{G2gError, HardwareError, PipelinePacket};

use crate::remotewire::map_wire;

/// Any WebSocket / transport failure maps to an internal hardware error (the
/// network boundary), matching how the CPU sinks treat a transport failure.
pub(crate) fn ws_err(_: WsError) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// Serialize `packet` with the wire codec and send it as one binary message.
pub(crate) async fn send_wire<S>(
    ws: &mut WebSocketStream<S>,
    packet: &PipelinePacket,
) -> Result<(), G2gError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let body: Vec<u8> = encode_packet(packet).map_err(map_wire)?;
    ws.send(Message::Binary(body)).await.map_err(ws_err)
}

/// Read the next wire packet, skipping WebSocket control / text frames (this
/// protocol only ever sends binary). `Ok(None)` on a clean WebSocket close (the
/// stream's natural end).
pub(crate) async fn recv_wire<S>(
    ws: &mut WebSocketStream<S>,
) -> Result<Option<PipelinePacket>, G2gError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    loop {
        match ws.next().await {
            Some(Ok(Message::Binary(bytes))) => {
                return Ok(Some(decode_packet(&bytes).map_err(map_wire)?));
            }
            // A clean close ends the stream; any non-binary frame (ping / pong /
            // text) is skipped (this protocol never sends them).
            Some(Ok(Message::Close(_))) | None => return Ok(None),
            Some(Ok(_)) => continue,
            Some(Err(e)) => return Err(ws_err(e)),
        }
    }
}
