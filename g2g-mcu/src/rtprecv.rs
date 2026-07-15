//! RTP ingress seam: the receive-direction inverse of [`RtpSink`](crate::RtpSink).
//! [`PacketReceiver`] is the thin trait a board's network stack implements (one
//! datagram in per call, the lwIP `recv` / Zephyr `recvfrom` shape); [`RtpSrc`]
//! is the heap-free [`StaticSource`] that receives a datagram, parses its RTP
//! header with the shared [`RtpHeader::parse`], and lends the payload downstream
//! zero-copy as a [`Frame`] whose `sequence` is the RTP sequence number (so a
//! [`JitterBuffer`](crate::JitterBuffer) can reorder it) and whose PTS is the RTP
//! timestamp mapped through the payload's [`MediaClock`].
//!
//! Like the rest of `g2g-mcu` it is host-testable: a mock receiver that returns
//! canned datagrams exercises the real parse + lend path, and the RX flagship
//! (`RtpSrc -> JitterBuffer -> G.711 decode -> PCM`) runs on mock peripherals.

use g2g_core::error::G2gError;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::mediaclock::MediaClock;
use g2g_core::rtp::RtpHeader;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticSource;

use crate::lend::lend_slot;

/// One received RTP datagram, handed over by the board's network stack.
///
/// `recv` fills `buf` with one datagram and returns its length (`0..=buf.len()`);
/// blocking until one arrives is the natural receive pacing (the stack wakes the
/// task), the mirror of [`PacketSender`](crate::PacketSender) blocking on send
/// back-pressure. A datagram longer than `buf` is the caller's MTU-sizing bug.
#[allow(async_fn_in_trait)]
pub trait PacketReceiver {
    /// Receive one datagram into `buf`, returning its byte length.
    async fn recv(&mut self, buf: &mut [u8]) -> Result<usize, G2gError>;

    /// Re-initialize the network stack after a fault (re-open / re-bind the
    /// socket), so the next `recv` starts clean. The default is a no-op; the
    /// supervisor's [`Recovery::Reset`](g2g_core::supervise::Recovery::Reset)
    /// invokes it via [`RtpSrc`]'s [`Recover`](g2g_core::supervise::Recover).
    async fn reset(&mut self) -> Result<(), G2gError> {
        Ok(())
    }
}

/// A heap-free RTP receive [`StaticSource`]: receive a datagram into a fixed
/// scratch buffer, parse the RTP header, and lend the payload from a
/// [`StaticLendRing`] as a [`Frame`]. `SLOTS` / `PAYLOAD` size the lend ring
/// (the payload, not the whole datagram), and `DGRAM` sizes the receive scratch
/// (at least `RTP_HEADER_LEN` + the largest payload, plus room for any CSRC /
/// extension a sender adds).
///
/// A datagram that fails to parse (not RTP v2, truncated) or that a payload-type
/// filter rejects is skipped and counted, not surfaced as an error: a receiver
/// must tolerate stray packets on the wire. A payload larger than `PAYLOAD` is a
/// ring-sizing bug surfaced as [`G2gError::CapsMismatch`].
pub struct RtpSrc<'r, R, const SLOTS: usize, const PAYLOAD: usize, const DGRAM: usize> {
    receiver: R,
    ring: &'r StaticLendRing<SLOTS, PAYLOAD>,
    clock: MediaClock,
    scratch: [u8; DGRAM],
    expected_pt: Option<u8>,
    remaining: Option<u32>,
    /// Datagrams dropped because they did not parse as RTP.
    malformed: u32,
    /// Datagrams dropped by the payload-type filter.
    filtered: u32,
}

impl<R, const SLOTS: usize, const PAYLOAD: usize, const DGRAM: usize>
    RtpSrc<'static, R, SLOTS, PAYLOAD, DGRAM>
where
    R: PacketReceiver,
{
    /// An RTP source over a `'static` ring (the MCU idiom), which makes the
    /// zero-copy lend sound by construction. `clock` is the payload's media
    /// clock (e.g. `MediaClock::audio(8000)` for G.711), used to derive each
    /// frame's PTS from the RTP timestamp.
    pub fn new(
        receiver: R,
        ring: &'static StaticLendRing<SLOTS, PAYLOAD>,
        clock: MediaClock,
    ) -> Self {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(receiver, ring, clock) }
    }
}

impl<'r, R, const SLOTS: usize, const PAYLOAD: usize, const DGRAM: usize>
    RtpSrc<'r, R, SLOTS, PAYLOAD, DGRAM>
where
    R: PacketReceiver,
{
    /// An RTP source over a borrowed ring.
    ///
    /// # Safety
    /// The ring must outlive every frame this source publishes (the
    /// [`RingSlot::publish`](g2g_core::staticpool::RingSlot::publish) contract).
    pub unsafe fn with_ring(
        receiver: R,
        ring: &'r StaticLendRing<SLOTS, PAYLOAD>,
        clock: MediaClock,
    ) -> Self {
        Self {
            receiver,
            ring,
            clock,
            scratch: [0u8; DGRAM],
            expected_pt: None,
            remaining: None,
            malformed: 0,
            filtered: 0,
        }
    }

    /// Only accept datagrams with this RTP payload type (e.g. 0 for PCMU);
    /// others are dropped and counted. Without it, every parsable RTP packet is
    /// accepted.
    pub fn with_payload_type(mut self, pt: u8) -> Self {
        self.expected_pt = Some(pt & 0x7F);
        self
    }

    /// End the stream after `frames` accepted packets (a receiver is endless by
    /// default; proofs and tests bound it).
    pub fn with_frame_limit(mut self, frames: u32) -> Self {
        self.remaining = Some(frames);
        self
    }

    /// Datagrams dropped because they did not parse as RTP.
    pub fn malformed(&self) -> u32 {
        self.malformed
    }

    /// Datagrams dropped by the payload-type filter.
    pub fn filtered(&self) -> u32 {
        self.filtered
    }

    /// Release the receiver.
    pub fn free(self) -> R {
        self.receiver
    }
}

impl<R, const SLOTS: usize, const PAYLOAD: usize, const DGRAM: usize> StaticSource
    for RtpSrc<'_, R, SLOTS, PAYLOAD, DGRAM>
where
    R: PacketReceiver,
{
    async fn next(&mut self) -> Result<Option<Frame>, G2gError> {
        if self.remaining == Some(0) {
            return Ok(None);
        }
        let ring = self.ring; // copy the reference out so the fill closure can borrow scratch
        loop {
            let len = self.receiver.recv(&mut self.scratch).await?;
            let Some(datagram) = self.scratch.get(..len) else {
                self.malformed = self.malformed.wrapping_add(1);
                continue;
            };
            let Some(parsed) = RtpHeader::parse(datagram) else {
                self.malformed = self.malformed.wrapping_add(1);
                continue;
            };
            if let Some(pt) = self.expected_pt {
                if parsed.header.payload_type != pt {
                    self.filtered = self.filtered.wrapping_add(1);
                    continue;
                }
            }
            if parsed.payload_len == 0 {
                self.malformed = self.malformed.wrapping_add(1);
                continue;
            }
            if parsed.payload_len > PAYLOAD {
                return Err(G2gError::CapsMismatch);
            }
            let Some(payload) =
                self.scratch.get(parsed.payload_offset..parsed.payload_offset + parsed.payload_len)
            else {
                self.malformed = self.malformed.wrapping_add(1);
                continue;
            };
            // PTS from the RTP timestamp through the payload media clock (the
            // sender's clock, so it wraps at 32 bits; the reorder key is the
            // sequence number, carried in `Frame::sequence`).
            let pts_ns = self.clock.ticks_to_ns(parsed.header.timestamp as u64);
            let timing = FrameTiming { pts_ns, ..FrameTiming::default() };
            // SAFETY: the constructor established the ring-outlives-frames
            // contract (`new`: 'static; `with_ring`: caller's contract).
            let frame = unsafe {
                lend_slot(ring, timing, parsed.header.sequence as u64, parsed.payload_len, |dst| {
                    dst.copy_from_slice(payload);
                })?
            };
            if let Some(remaining) = &mut self.remaining {
                *remaining -= 1;
            }
            return Ok(Some(frame));
        }
    }
}

impl<R, const SLOTS: usize, const PAYLOAD: usize, const DGRAM: usize> g2g_core::supervise::Recover
    for RtpSrc<'_, R, SLOTS, PAYLOAD, DGRAM>
where
    R: PacketReceiver,
{
    /// Recover the RTP source by re-initializing its [`PacketReceiver`]
    /// (re-open the socket after a network fault).
    async fn recover(&mut self) -> Result<(), G2gError> {
        self.receiver.reset().await
    }
}

impl<R, const SLOTS: usize, const PAYLOAD: usize, const DGRAM: usize> core::fmt::Debug
    for RtpSrc<'_, R, SLOTS, PAYLOAD, DGRAM>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RtpSrc")
            .field("slots", &SLOTS)
            .field("payload_bytes", &PAYLOAD)
            .field("dgram_bytes", &DGRAM)
            .field("expected_pt", &self.expected_pt)
            .field("malformed", &self.malformed)
            .field("filtered", &self.filtered)
            .finish_non_exhaustive()
    }
}
