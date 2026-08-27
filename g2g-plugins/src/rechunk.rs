//! Byte-stream re-chunker shared by the `rndbuffersize` and `chopmydata` debug
//! transforms. Both hold input bytes back until the size they drew is available,
//! cut that many off the front and push them as one buffer; only the size
//! formula differs, so it stays with the caller.

use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket};

/// Bytes waiting to be cut, plus the timing the cut buffers inherit.
#[derive(Debug, Default)]
pub(crate) struct Rechunker {
    /// Input bytes not yet long enough for the next chunk.
    pending: Vec<u8>,
    /// The size drawn for the next chunk, held while its bytes are still
    /// arriving so one draw is spent per emitted buffer.
    next_target: Option<usize>,
    /// pts of the input buffer being consumed, carried by the chunks cut from it.
    pts_ns: u64,
    sequence: u64,
}

impl Rechunker {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Buffers emitted so far.
    pub(crate) fn emitted(&self) -> u64 {
        self.sequence
    }

    /// Take one input buffer's bytes; its pts stamps every chunk cut from here on.
    pub(crate) fn accept(&mut self, bytes: &[u8], pts_ns: u64) {
        self.pending.extend_from_slice(bytes);
        self.pts_ns = pts_ns;
    }

    /// Drop the bytes held back and the size drawn for them. A flushing seek
    /// restarts the stream, so a half-filled chunk is bytes downstream must
    /// never see joined to what comes next.
    pub(crate) fn clear(&mut self) {
        self.pending.clear();
        self.next_target = None;
    }

    /// Forget the size drawn for the next chunk, keeping the bytes. The next
    /// drain spends a fresh draw, which is how a caller switches cut sizes
    /// mid-stream (`chopmydata` cuts its tail at `min-size`).
    pub(crate) fn clear_target(&mut self) {
        self.next_target = None;
    }

    /// Everything still held back, leaving the rechunker empty.
    pub(crate) fn take_pending(&mut self) -> Vec<u8> {
        self.next_target = None;
        core::mem::take(&mut self.pending)
    }

    /// Cut every whole chunk the pending bytes hold and push it, spending one
    /// `next_size` draw per emitted buffer.
    pub(crate) async fn drain(
        &mut self,
        mut next_size: impl FnMut() -> usize,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        loop {
            let size = match self.next_target {
                Some(size) => size,
                None => *self.next_target.insert(next_size()),
            };
            if self.pending.len() < size {
                return Ok(());
            }
            self.next_target = None;
            let chunk: Vec<u8> = self.pending.drain(..size).collect();
            self.emit(chunk, out).await?;
        }
    }

    /// Push `bytes` as one buffer, stamped with the pts of the input it came from.
    pub(crate) async fn emit(
        &mut self,
        bytes: Vec<u8>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming {
                pts_ns: self.pts_ns,
                dts_ns: self.pts_ns,
                ..FrameTiming::default()
            },
            self.sequence,
        );
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}
