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

    use g2g_core::wire::{decode_packet, encode_packet, WireError};
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

    /// Reader for the length framing, holding the partly-read frame across calls.
    /// That is what makes it cancel-safe: the WebTransport carrier polls it in a
    /// `select!` against the session's datagram flow, so the read future is
    /// dropped whenever a datagram wins the race, and the bytes already off the
    /// stream must survive that.
    #[derive(Debug, Default)]
    pub(crate) struct FramedRead {
        /// The current frame so far: the 4-byte length prefix, then its body.
        buf: Vec<u8>,
    }

    impl FramedRead {
        /// Read until `buf` holds `target` bytes; `false` on a clean end. Never
        /// reads past `target`, so nothing of a later frame is left behind, and
        /// the buffer only grows by what actually arrived.
        async fn fill_to<R: AsyncRead + Unpin>(
            &mut self,
            src: &mut R,
            target: usize,
        ) -> Result<bool, G2gError> {
            let mut chunk = [0u8; 8 * 1024];
            while self.buf.len() < target {
                let want = (target - self.buf.len()).min(chunk.len());
                // A single `read` is cancel-safe: dropped before it resolves, it
                // has taken nothing off the stream.
                let n = src.read(&mut chunk[..want]).await.map_err(io_err)?;
                if n == 0 {
                    return Ok(false);
                }
                self.buf.extend_from_slice(&chunk[..n]);
            }
            Ok(true)
        }

        /// Read the next length-framed packet, decoded. `Ok(None)` on a clean
        /// close at a frame boundary (the stream's natural end).
        pub(crate) async fn recv<R: AsyncRead + Unpin>(
            &mut self,
            src: &mut R,
        ) -> Result<Option<PipelinePacket>, G2gError> {
            if !self.fill_to(src, 4).await? {
                return Ok(None);
            }
            // The length is peer-supplied: nothing is sized on it up front, and a
            // stream that ends short of it fails as a truncation.
            let n =
                u32::from_le_bytes([self.buf[0], self.buf[1], self.buf[2], self.buf[3]]) as usize;
            if !self.fill_to(src, n.saturating_add(4)).await? {
                return Err(map_wire(WireError::Truncated));
            }
            let packet = decode_packet(&self.buf[4..]).map_err(map_wire)?;
            self.buf.clear();
            Ok(Some(packet))
        }
    }

    /// Read the next length-framed packet from a stream nothing else reads.
    #[cfg(feature = "remote")]
    pub(crate) async fn recv_framed<R: AsyncRead + Unpin>(
        src: &mut R,
    ) -> Result<Option<PipelinePacket>, G2gError> {
        FramedRead::default().recv(src).await
    }
}

#[cfg(feature = "remote")]
pub(crate) use framed::recv_framed;
#[cfg(any(feature = "remote", feature = "webtransport"))]
pub(crate) use framed::send_framed;
#[cfg(feature = "webtransport")]
pub(crate) use framed::FramedRead;
