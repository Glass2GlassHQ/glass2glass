//! RTP egress seam (M643): the [`PacketSender`] contract the board's network
//! stack implements, plus [`RtpSink`], the heap-free [`StaticSink`] that
//! turns each audio frame into one RTP packet, the last stage of the
//! reference audio chain (capture -> convert -> resample -> mix -> encode ->
//! RTP). The element owns everything portable (the RFC 3550 header via
//! [`g2g_core::rtp::RtpHeader`], the one shared implementation, timestamping
//! through [`MediaClock`], sequence numbering, size validation); a board
//! supplies only the sender, an adapter over its UDP stack (lwIP `pbuf`,
//! Zephyr `sendmsg`, smoltcp), which is also where the host validation plugs
//! a real socket to face ffmpeg as the receiving peer.
//!
//! One frame is one packet on purpose: a deterministic MCU chain sizes its
//! frames to the MTU by construction (20 ms of G.711 at 8 kHz is 160 bytes),
//! so an oversized payload is a configuration bug surfaced as
//! [`G2gError::CapsMismatch`], not silently fragmented. Formats that need
//! fragmentation rules (H.264 FU-A) have format-specific packetizers on the
//! std side (`g2g-plugins::rtppay`); byte-preserving audio payloads (G.711,
//! ADPCM, L16) need none.

use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::memory::MemoryDomain;
use g2g_core::rtp::{RtpHeader, RTP_HEADER_LEN};
use g2g_core::time::TaiNs;
use g2g_core::{MediaClock, StaticSink};

/// One RTP packet handed to the board's network stack as header + payload,
/// split so a scatter-gather send (an lwIP `pbuf` chain, a `sendmsg` iovec)
/// stays zero-copy; a flat-buffer stack concatenates the two parts itself.
///
/// Blocking until the stack can take the datagram is the pipeline's natural
/// network back-pressure (like [`PcmWriter`](crate::PcmWriter) on the audio
/// side); dropping instead is the adapter's policy to make.
#[allow(async_fn_in_trait)]
pub trait PacketSender {
    /// Send one datagram consisting of `header` immediately followed by
    /// `payload`.
    async fn send(&mut self, header: &[u8; RTP_HEADER_LEN], payload: &[u8]) -> Result<(), G2gError>;

    /// Re-initialize the network stack after a fault (re-open the socket,
    /// re-bind, clear a stuck send queue), so the next `send` starts clean. The
    /// default is a no-op; the supervisor's
    /// [`Recovery::Reset`](g2g_core::supervise::Recovery::Reset) invokes it via
    /// [`RtpSink`]'s [`Recover`](g2g_core::supervise::Recover) impl.
    async fn reset(&mut self) -> Result<(), G2gError> {
        Ok(())
    }
}

/// A heap-free [`StaticSink`] emitting one RTP packet per input frame
/// through a [`PacketSender`].
///
/// The RTP timestamp is the frame's PTS mapped through the payload's
/// [`MediaClock`] (for the audio chain: the sample clock, so a PTS advancing
/// by real time advances the timestamp by the sample count, exactly what a
/// reference receiver expects). The marker bit is set on the first packet
/// only (RFC 3551: the start of a talkspurt after silence; a continuous
/// stream has exactly one). Sequence numbers start at the constructor's
/// value and wrap.
pub struct RtpSink<S> {
    sender: S,
    clock: MediaClock,
    payload_type: u8,
    ssrc: u32,
    sequence: u16,
    /// Max payload bytes per packet (the MTU budget after the fixed header).
    max_payload: usize,
    started: bool,
}

impl<S: PacketSender> RtpSink<S> {
    /// A sink stamping `payload_type` packets on `clock` (the payload's RTP
    /// media clock, e.g. `MediaClock::audio(8000)` for G.711), with the given
    /// SSRC and initial sequence number (an MCU picks both at boot; there is
    /// no entropy source to draw on down here). Payloads are capped at 1400
    /// bytes; see [`with_max_payload`](Self::with_max_payload).
    pub fn new(sender: S, clock: MediaClock, payload_type: u8, ssrc: u32, sequence: u16) -> Self {
        Self {
            sender,
            clock,
            payload_type: payload_type & 0x7F,
            ssrc,
            sequence,
            max_payload: 1400,
            started: false,
        }
    }

    /// Max payload bytes per packet (a link with a smaller MTU budget).
    /// Frames larger than this are rejected, never fragmented.
    pub fn with_max_payload(mut self, bytes: usize) -> Self {
        self.max_payload = bytes;
        self
    }

    /// The sequence number the next packet will carry.
    pub fn next_sequence(&self) -> u16 {
        self.sequence
    }

    /// Release the sender (e.g. to close the socket).
    pub fn free(self) -> S {
        self.sender
    }
}

impl<S: PacketSender> StaticSink for RtpSink<S> {
    async fn consume(&mut self, frame: Frame) -> Result<(), G2gError> {
        let MemoryDomain::System(slice) = &frame.domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        let payload = slice.as_slice();
        // One frame is one packet: an empty or over-MTU payload is a
        // configuration bug upstream, not something to paper over here.
        if payload.is_empty() || payload.len() > self.max_payload {
            return Err(G2gError::CapsMismatch);
        }
        let header = RtpHeader {
            payload_type: self.payload_type,
            marker: !self.started,
            sequence: self.sequence,
            // The board's monotonic capture clock in media ticks, mod 2^32:
            // the same PTS -> RTP mapping the ST 2110 elements use.
            timestamp: self.clock.rtp_timestamp(TaiNs(frame.timing.pts_ns)).get(),
            ssrc: self.ssrc,
        };
        self.sender.send(&header.to_bytes(), payload).await?;
        self.started = true;
        self.sequence = self.sequence.wrapping_add(1);
        Ok(())
    }
}

impl<S: PacketSender> g2g_core::supervise::Recover for RtpSink<S> {
    /// Recover the egress sink by re-initializing its [`PacketSender`]
    /// (`reset`), re-opening the transport after a network fault. Sequence and
    /// SSRC continue unchanged so a receiver's stream is not disrupted by the
    /// recovery.
    async fn recover(&mut self) -> Result<(), G2gError> {
        self.sender.reset().await
    }
}

impl<S> core::fmt::Debug for RtpSink<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RtpSink")
            .field("payload_type", &self.payload_type)
            .field("ssrc", &self.ssrc)
            .field("sequence", &self.sequence)
            .finish_non_exhaustive()
    }
}
