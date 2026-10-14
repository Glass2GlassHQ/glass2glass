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
        _ => G2gError::Hardware(HardwareError::Other),
    }
}

/// Length framing for the byte-stream transports (TCP, and a WebTransport
/// bidirectional stream): neither delimits messages, so one wire body is a `u32`
/// LE length followed by that many bytes. WebSocket needs none of this (it is
/// already message-framed).
#[cfg(any(feature = "remote", feature = "webtransport"))]
mod framed {
    use alloc::vec::Vec;
    use std::io::{self, IoSlice};

    use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

    use g2g_core::wire::{decode_packet, encode_packet, WireError};
    use g2g_core::{G2gError, PipelinePacket};

    use crate::filesink::io_err;
    use crate::remotewire::map_wire;

    const LENGTH_PREFIX_LEN: usize = 4;
    // bounds what a peer's claimed length can make us allocate before its bytes arrive
    const MAX_READ_RESERVE: usize = 1 << 20;

    /// Serialize `packet` and write it length-framed.
    pub(crate) async fn send_framed<W: AsyncWrite + Unpin>(
        sink: &mut W,
        packet: &PipelinePacket,
    ) -> Result<(), G2gError> {
        let body: Vec<u8> = encode_packet(packet).map_err(map_wire)?;
        let length = (body.len() as u32).to_le_bytes();
        // prefix and body in one write, two writes let Nagle hold the body back
        let mut written = 0;
        while written < LENGTH_PREFIX_LEN {
            let parts = [IoSlice::new(&length[written..]), IoSlice::new(&body)];
            let n = sink.write_vectored(&parts).await.map_err(io_err)?;
            if n == 0 {
                return Err(io_err(io::ErrorKind::WriteZero.into()));
            }
            written += n;
        }
        sink.write_all(&body[written - LENGTH_PREFIX_LEN..])
            .await
            .map_err(io_err)?;
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
            while self.buf.len() < target {
                let remaining = target - self.buf.len();
                self.buf.reserve(remaining.min(MAX_READ_RESERVE));
                // A single `read_buf` is cancel-safe: dropped before it resolves,
                // it has taken nothing off the stream.
                let n = (&mut *src)
                    .take(remaining as u64)
                    .read_buf(&mut self.buf)
                    .await
                    .map_err(io_err)?;
                if n == 0 {
                    return Ok(false);
                }
            }
            Ok(true)
        }

        /// Read the next length-framed packet, decoded. `Ok(None)` on a clean
        /// close at a frame boundary (the stream's natural end).
        pub(crate) async fn recv<R: AsyncRead + Unpin>(
            &mut self,
            src: &mut R,
        ) -> Result<Option<PipelinePacket>, G2gError> {
            if !self.fill_to(src, LENGTH_PREFIX_LEN).await? {
                return Ok(None);
            }
            // The length is peer-supplied: nothing is sized on it up front, and a
            // stream that ends short of it fails as a truncation.
            let n =
                u32::from_le_bytes([self.buf[0], self.buf[1], self.buf[2], self.buf[3]]) as usize;
            if !self
                .fill_to(src, n.saturating_add(LENGTH_PREFIX_LEN))
                .await?
            {
                return Err(map_wire(WireError::Truncated));
            }
            let packet = decode_packet(&self.buf[LENGTH_PREFIX_LEN..]).map_err(map_wire)?;
            self.buf.clear();
            Ok(Some(packet))
        }
    }
}

#[cfg(any(feature = "remote", feature = "webtransport"))]
pub(crate) use framed::{send_framed, FramedRead};

#[cfg(all(test, any(feature = "remote", feature = "webtransport")))]
mod tests {
    use super::*;
    use g2g_core::frame::Frame;
    use g2g_core::memory::{MemoryDomain, SystemSlice};
    use g2g_core::{FrameTiming, PipelinePacket};

    // forces partial writes and reads, so every frame boundary is crossed mid-call
    const PIPE_CAPACITY: usize = 3;

    fn frame(len: usize, sequence: u64) -> PipelinePacket {
        let bytes: alloc::vec::Vec<u8> = (0..len).map(|i| i as u8).collect();
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence,
            meta: Default::default(),
        })
    }

    #[tokio::test]
    async fn framed_packets_survive_partial_writes_and_reads() {
        const FRAME_LENS: [usize; 3] = [0, 1, 5000];
        let (mut writer, mut reader) = tokio::io::duplex(PIPE_CAPACITY);
        let sending = tokio::spawn(async move {
            for (sequence, len) in FRAME_LENS.into_iter().enumerate() {
                send_framed(&mut writer, &frame(len, sequence as u64))
                    .await
                    .unwrap();
            }
            send_framed(&mut writer, &PipelinePacket::Eos)
                .await
                .unwrap();
        });
        let mut framed = FramedRead::default();
        for (sequence, len) in FRAME_LENS.into_iter().enumerate() {
            let Some(PipelinePacket::DataFrame(received)) = framed.recv(&mut reader).await.unwrap()
            else {
                panic!("frame {sequence} missing");
            };
            assert_eq!(received.sequence, sequence as u64);
            let MemoryDomain::System(bytes) = &received.domain else {
                panic!("system memory frame");
            };
            let expected: alloc::vec::Vec<u8> = (0..len).map(|i| i as u8).collect();
            assert_eq!(bytes.as_slice(), expected.as_slice());
        }
        assert!(matches!(
            framed.recv(&mut reader).await.unwrap(),
            Some(PipelinePacket::Eos)
        ));
        assert!(framed.recv(&mut reader).await.unwrap().is_none());
        sending.await.unwrap();
    }
}
