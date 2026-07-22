//! UART transport seams and elements: the byte-stream egress / ingress a real
//! product uses for telemetry, a console, or a serial sensor, alongside the
//! packet (RTP) and bus (SPI / I2C / I2S) transports. `embedded-hal` 1.0 keeps
//! blocking serial in the separate `embedded-io` crate, so, as with
//! [`PacketSender`](crate::PacketSender) / [`PacketReceiver`](crate::PacketReceiver),
//! these are local one-method seams a vendor HAL's UART satisfies with a trivial
//! adapter, keeping the elements host-testable with a mock link.
//!
//! [`UartSink`] writes each frame's payload over the TX seam (a raw byte egress;
//! the producer chooses any framing). [`UartSrc`] reads fixed-size frames from
//! the RX seam, the receive counterpart, so the two round-trip over a link.

use g2g_core::error::G2gError;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::MemoryDomain;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{StaticSink, StaticSource};

/// The transmit side of a UART: write a slice of bytes, blocking until the
/// hardware FIFO / DMA has taken them (the natural egress back-pressure, like
/// [`PacketSender`](crate::PacketSender)).
#[allow(async_fn_in_trait)]
pub trait SerialTx {
    /// Write all of `bytes`.
    async fn write_all(&mut self, bytes: &[u8]) -> Result<(), G2gError>;

    /// Re-initialize the UART after a fault (clear an overrun / framing error,
    /// re-enable the transmitter). Default no-op; the supervisor's
    /// [`Recovery::Reset`](g2g_core::supervise::Recovery::Reset) invokes it.
    async fn reset(&mut self) -> Result<(), G2gError> {
        Ok(())
    }
}

/// The receive side of a UART: read into `buf`, returning the byte count. A
/// return of `0` signals end of stream (the link closed), which [`UartSrc`]
/// treats as a clean end only at a frame boundary.
#[allow(async_fn_in_trait)]
pub trait SerialRx {
    /// Read up to `buf.len()` bytes into `buf`, returning how many (0 = closed).
    async fn read(&mut self, buf: &mut [u8]) -> Result<usize, G2gError>;

    /// Re-initialize the UART after a fault. Default no-op.
    async fn reset(&mut self) -> Result<(), G2gError> {
        Ok(())
    }
}

/// A heap-free UART egress [`StaticSink`]: writes each frame's payload verbatim
/// over a [`SerialTx`] (telemetry / console output). Framing, if any, is the
/// upstream producer's choice.
pub struct UartSink<T> {
    tx: T,
}

impl<T: SerialTx> UartSink<T> {
    /// A UART sink over a transmit seam.
    pub fn new(tx: T) -> Self {
        Self { tx }
    }

    /// Release the transmit seam.
    pub fn free(self) -> T {
        self.tx
    }
}

impl<T: SerialTx> StaticSink for UartSink<T> {
    async fn consume(&mut self, frame: Frame) -> Result<(), G2gError> {
        let Some(slice) = frame.domain.as_system_slice() else {
            return Err(G2gError::UnsupportedDomain);
        };
        self.tx.write_all(slice).await
    }
}

impl<T: SerialTx> g2g_core::supervise::Recover for UartSink<T> {
    async fn recover(&mut self) -> Result<(), G2gError> {
        self.tx.reset().await
    }
}

impl<T> core::fmt::Debug for UartSink<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("UartSink").finish_non_exhaustive()
    }
}

/// A heap-free UART ingress [`StaticSource`]: reads fixed-size `frame_len`-byte
/// frames from a [`SerialRx`] and lends each downstream zero-copy from a
/// [`StaticLendRing`]. A short read is retried until a whole frame is assembled;
/// end of stream at a frame boundary ends cleanly, mid-frame is a fault.
pub struct UartSrc<'r, R, const N: usize, const BYTES: usize> {
    rx: R,
    ring: &'r StaticLendRing<N, BYTES>,
    frame_len: usize,
    frame_interval_ns: u64,
    remaining: Option<u32>,
    seq: u64,
}

impl<R: SerialRx, const N: usize, const BYTES: usize> UartSrc<'static, R, N, BYTES> {
    /// A UART source reading `frame_len`-byte frames over a `'static` ring
    /// (the MCU idiom). `frame_len` must be `<= BYTES`.
    pub fn new(
        rx: R,
        ring: &'static StaticLendRing<N, BYTES>,
        frame_len: usize,
        frame_interval_ns: u64,
    ) -> Self {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(rx, ring, frame_len, frame_interval_ns) }
    }
}

impl<'r, R: SerialRx, const N: usize, const BYTES: usize> UartSrc<'r, R, N, BYTES> {
    /// A UART source over a borrowed ring.
    ///
    /// # Safety
    /// The ring must outlive every frame this source publishes (the
    /// [`RingSlot::publish`](g2g_core::staticpool::RingSlot::publish) contract).
    pub unsafe fn with_ring(
        rx: R,
        ring: &'r StaticLendRing<N, BYTES>,
        frame_len: usize,
        frame_interval_ns: u64,
    ) -> Self {
        Self {
            rx,
            ring,
            frame_len,
            frame_interval_ns,
            remaining: None,
            seq: 0,
        }
    }

    /// End the stream after `frames` frames.
    pub fn with_frame_limit(mut self, frames: u32) -> Self {
        self.remaining = Some(frames);
        self
    }

    /// Release the receive seam.
    pub fn free(self) -> R {
        self.rx
    }
}

impl<R: SerialRx, const N: usize, const BYTES: usize> StaticSource for UartSrc<'_, R, N, BYTES> {
    async fn next(&mut self) -> Result<Option<Frame>, G2gError> {
        if self.remaining == Some(0) {
            return Ok(None);
        }
        let want = self.frame_len.min(BYTES);
        let Some(mut slot) = self.ring.acquire() else {
            return Err(G2gError::PoolExhausted);
        };
        let mut got = 0;
        while got < want {
            let Some(dst) = slot.buf_mut().get_mut(got..want) else {
                return Err(G2gError::CapsMismatch);
            };
            let n = self.rx.read(dst).await?;
            if n == 0 {
                // End of stream: clean only at a frame boundary.
                if got == 0 {
                    return Ok(None);
                }
                return Err(G2gError::CapsMismatch);
            }
            got = got.saturating_add(n).min(want);
        }
        let pts_ns = self.seq.saturating_mul(self.frame_interval_ns);
        // SAFETY: the constructor established the ring-outlives-frames contract
        // (`new`: 'static; `with_ring`: caller's contract).
        let slice = unsafe { slot.publish(want) };
        let frame = Frame::new(
            MemoryDomain::System(slice),
            FrameTiming {
                pts_ns,
                ..FrameTiming::default()
            },
            self.seq,
        );
        if let Some(remaining) = &mut self.remaining {
            *remaining -= 1;
        }
        self.seq += 1;
        Ok(Some(frame))
    }
}

impl<R: SerialRx, const N: usize, const BYTES: usize> g2g_core::supervise::Recover
    for UartSrc<'_, R, N, BYTES>
{
    async fn recover(&mut self) -> Result<(), G2gError> {
        self.rx.reset().await
    }
}

impl<R, const N: usize, const BYTES: usize> core::fmt::Debug for UartSrc<'_, R, N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("UartSrc")
            .field("frame_len", &self.frame_len)
            .field("slots", &N)
            .field("slot_bytes", &BYTES)
            .field("seq", &self.seq)
            .finish_non_exhaustive()
    }
}
