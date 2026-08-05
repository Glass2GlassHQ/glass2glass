//! Shared helpers for the distributed-graph transport elements (the `remote` TCP
//! pair, the `remote-ws` WebSocket pair and the `webtransport` QUIC family):
//! mapping a [`WireError`] from the `g2g-core` codec to the pipeline error type,
//! and the length-framed read / write the byte-stream transports share. All three
//! serialize the identical `PipelinePacket` stream through [`g2g_core::wire`];
//! only the byte transport under it differs.

use g2g_core::wire::WireError;
use g2g_core::{G2gError, HardwareError};

/// Map a [`WireError`] to the pipeline error type. A device / foreign memory
/// domain surfaces as `UnsupportedDomain` (the same error a CPU sink raises);
/// anything else is an internal encode / decode fault.
pub(crate) fn map_wire(e: WireError) -> G2gError {
    match e {
        WireError::UnsupportedDomain => G2gError::UnsupportedDomain,
        WireError::Truncated | WireError::BadTag => G2gError::Hardware(HardwareError::Other),
    }
}

/// Length framing for the byte-stream transports (TCP, and a WebTransport
/// bidirectional stream): neither delimits messages, so one wire body is a `u32`
/// LE length followed by that many bytes. WebSocket needs none of this (it is
/// already message-framed).
#[cfg(any(feature = "remote", feature = "webtransport"))]
mod framed {
    use alloc::vec::Vec;

    use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

    use g2g_core::wire::{decode_packet, encode_packet};
    use g2g_core::{G2gError, PipelinePacket};

    use crate::filesink::io_err;
    use crate::remotewire::map_wire;

    /// Serialize `packet` and write it length-framed.
    pub(crate) async fn send_framed<W: AsyncWrite + Unpin>(
        sink: &mut W,
        packet: &PipelinePacket,
    ) -> Result<(), G2gError> {
        let body: Vec<u8> = encode_packet(packet).map_err(map_wire)?;
        sink.write_all(&(body.len() as u32).to_le_bytes())
            .await
            .map_err(io_err)?;
        sink.write_all(&body).await.map_err(io_err)?;
        Ok(())
    }

    /// Read one length-framed wire body. `Ok(None)` on a clean close at a frame
    /// boundary (the stream's natural end).
    async fn read_framed<R: AsyncRead + Unpin>(src: &mut R) -> Result<Option<Vec<u8>>, G2gError> {
        let mut len = [0u8; 4];
        match src.read_exact(&mut len).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(io_err(e)),
        }
        // The length is peer-supplied, so a bogus one must not preallocate: read
        // it back in bounded chunks and let a short stream fail as a truncation.
        let n = u32::from_le_bytes(len) as usize;
        let mut body = Vec::new();
        let mut left = n;
        while left > 0 {
            let chunk = left.min(64 * 1024);
            let base = body.len();
            body.resize(base + chunk, 0u8);
            src.read_exact(&mut body[base..]).await.map_err(io_err)?;
            left -= chunk;
        }
        Ok(Some(body))
    }

    /// Read the next length-framed packet, decoded.
    pub(crate) async fn recv_framed<R: AsyncRead + Unpin>(
        src: &mut R,
    ) -> Result<Option<PipelinePacket>, G2gError> {
        match read_framed(src).await? {
            Some(body) => Ok(Some(decode_packet(&body).map_err(map_wire)?)),
            None => Ok(None),
        }
    }
}

#[cfg(any(feature = "remote", feature = "webtransport"))]
pub(crate) use framed::{recv_framed, send_framed};
