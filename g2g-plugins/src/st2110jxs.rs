//! ST 2110-22 JPEG XS over RTP (M604): the packetizer and depacketizer for
//! constant-bit-rate compressed video, RFC 9134 (ISO/IEC 21122 / JPEG XS).
//!
//! ST 2110-22 carries a compressed JPEG XS codestream where -20 carries raw
//! pixels: the same PTP-locked, per-frame-timestamped RTP transport, but a
//! visually lossless mezzanine codec at a fraction of -20's bandwidth. Unlike
//! -20 (a per-packet SRD line header naming where each pixel run lands) this is
//! a bytestream: the whole codestream of one frame is sliced into MTU-sized
//! packets, each prefixed with the RFC 9134 4-octet payload header, and the RTP
//! marker bit ends the frame. Every packet of one frame shares that frame's
//! [`MediaClock`] (90 kHz) timestamp, so a receiver on the same grandmaster
//! reconstructs the sampling instant, the video half of A/V sync across devices.
//!
//! We implement **codestream packetization mode** (RFC 9134 `K=0`, the ST 2110-22
//! norm): one packetization unit per video frame, packets carried in order
//! (transmission mode `T=1`). The 4-octet payload header is:
//!
//! ```text
//!  0                   1                   2                   3
//!  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |T|K|L| I |  F counter  |     SEP counter     |    P counter    |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! ```
//!
//! `T` transmission mode, `K` packetization mode, `L` last-packet-of-unit, `I`
//! interlace (00 = progressive), `F` frame number mod 32, `SEP` the high bits of
//! the packet counter (codestream mode) and `P` the packet number within the unit
//! (mod 2048); together `SEP:P` is a 22-bit packet index.
//!
//! Sans-IO core (a codestream <-> RTP packets), like `st2110audio` / `st2110video`:
//! pure `no_std` + alloc, CI round-trip testable, an element wrapper
//! (`st2110jxsrtp`) sits on top. The codestream itself is opaque here (the codec
//! lives in `ffmpegjpegxs`); this layer never parses it.
//!
//! **Never trust the stream:** a receiver bounds the reassembly buffer (a sender
//! that never sets the marker cannot grow it without limit), drops a packet whose
//! packetization mode it does not implement (`K=1` slice mode), and discards a
//! partial frame when a new timestamp arrives before the marker (a lost final
//! packet), rather than emitting a truncated or run-on codestream.

use alloc::vec::Vec;

use g2g_core::rtp::{RtpHeader, RTP_HEADER_LEN};
use g2g_core::MediaClock;

/// RFC 9134 payload header length (T|K|L|I|F|SEP|P, one 32-bit word).
const JXS_HEADER_LEN: usize = 4;

/// Pack the RFC 9134 4-octet payload header. `last` sets `L`; `packet_index` is the
/// packet number within the packetization unit, split into the 11-bit `P` counter
/// and the 11-bit `SEP` high bits (codestream mode); `frame` is the frame number
/// (masked to the 5-bit `F` counter). `T=1` (sequential), `K=0` (codestream), `I=0`
/// (progressive).
fn jxs_header(last: bool, packet_index: u32, frame: u8) -> [u8; 4] {
    let p = packet_index & 0x7FF; // low 11 bits
    let sep = (packet_index >> 11) & 0x7FF; // high 11 bits
    let word: u32 = (1 << 31) // T = 1 (sequential)
        // K = 0 (codestream mode), I = 0 (progressive)
        | (u32::from(last) << 29)
        | ((u32::from(frame) & 0x1F) << 22)
        | (sep << 11)
        | p;
    word.to_be_bytes()
}

/// Packetizes a JPEG XS codestream (one frame) into ST 2110-22 (RFC 9134) RTP
/// packets, codestream packetization mode.
#[derive(Debug)]
pub struct St2110JxsPacketizer {
    payload_type: u8,
    ssrc: u32,
    sequence: u16,
    /// Frame number mod 32 for the payload header's `F` counter.
    frame_counter: u8,
    clock: MediaClock,
    /// Maximum RTP packet size in octets (header + payload); the codestream is
    /// sliced to fit it.
    max_packet: usize,
}

impl St2110JxsPacketizer {
    /// A packetizer capping each RTP packet at `max_packet` octets (a typical
    /// 1500-octet MTU leaves ~1460 after IP/UDP; pass that). `payload_type` is the
    /// dynamic RTP PT, `ssrc` the stream source id.
    pub fn new(payload_type: u8, ssrc: u32, max_packet: usize) -> Self {
        // Floor the cap at a header plus one payload octet, else no data ever fits.
        let floor = RTP_HEADER_LEN + JXS_HEADER_LEN + 1;
        Self {
            payload_type: payload_type & 0x7f,
            ssrc,
            sequence: 0,
            frame_counter: 0,
            clock: MediaClock::video(),
            max_packet: max_packet.max(floor),
        }
    }

    /// The media clock (for recovering a frame's PTP time on the receive side).
    pub fn media_clock(&self) -> MediaClock {
        self.clock
    }

    /// Packetize one JPEG XS `codestream` (a single frame) sampled at PTP/TAI time
    /// `tai_ns`. Every packet shares the frame's 90 kHz RTP timestamp; the last
    /// carries the marker bit and the payload header's `L` flag. An empty codestream
    /// still emits one (empty) marker packet, so a frame slot is never dropped.
    pub fn packetize(&mut self, codestream: &[u8], tai_ns: u64) -> Vec<Vec<u8>> {
        let rtp_ts = self.clock.rtp_timestamp(g2g_core::TaiNs(tai_ns)).get();
        let capacity = self.max_packet - RTP_HEADER_LEN - JXS_HEADER_LEN;
        // Number of packets: at least one (an empty codestream still ticks a frame).
        let n = codestream.len().div_ceil(capacity).max(1);

        let mut packets = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * capacity;
            let end = (start + capacity).min(codestream.len());
            let chunk = &codestream[start..end];
            let last = i + 1 == n;

            let mut pkt = Vec::with_capacity(RTP_HEADER_LEN + JXS_HEADER_LEN + chunk.len());
            // RTP header: marker on the last packet of the frame.
            let header = RtpHeader {
                payload_type: self.payload_type,
                marker: last,
                sequence: self.sequence,
                timestamp: rtp_ts,
                ssrc: self.ssrc,
            };
            pkt.extend_from_slice(&header.to_bytes());
            // RFC 9134 payload header.
            pkt.extend_from_slice(&jxs_header(last, i as u32, self.frame_counter));
            pkt.extend_from_slice(chunk);
            packets.push(pkt);

            self.sequence = self.sequence.wrapping_add(1);
        }
        self.frame_counter = self.frame_counter.wrapping_add(1);
        packets
    }
}

/// A completed ST 2110-22 frame: its RTP timestamp and the reassembled JPEG XS
/// codestream (opaque bytes, to be handed to a JPEG XS decoder).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct St2110JxsFrame {
    pub rtp_timestamp: u32,
    pub codestream: Vec<u8>,
}

/// Reassembles ST 2110-22 (RFC 9134) codestream-mode RTP packets into whole
/// JPEG XS frames.
#[derive(Debug)]
pub struct St2110JxsDepacketizer {
    /// The frame under reassembly; payloads are concatenated in arrival order and
    /// the marker bit ends the frame.
    buf: Vec<u8>,
    /// The accumulating frame's RTP timestamp (all its packets share it).
    ts: Option<u32>,
    /// Upper bound on a reassembled codestream; a sender that never marks the frame
    /// end cannot grow the buffer past this (never trust the stream).
    max_frame_bytes: usize,
}

impl St2110JxsDepacketizer {
    /// A depacketizer bounding a single reassembled codestream to `max_frame_bytes`.
    /// A frame exceeding it (a missing marker, or a bogus stream) is dropped rather
    /// than accumulated without limit.
    pub fn new(max_frame_bytes: usize) -> Self {
        Self {
            buf: Vec::new(),
            ts: None,
            max_frame_bytes: max_frame_bytes.max(1),
        }
    }

    /// Feed one RTP packet. Appends its JPEG XS payload to the frame under
    /// reassembly and, on the marker bit (end of frame), returns the completed
    /// codestream. Returns `None` while a frame is incomplete or if the packet is
    /// malformed (too short, wrong RTP version, or slice packetization mode, which
    /// we do not implement), in which case the packet is dropped. A new timestamp
    /// arriving mid-frame discards the (incomplete) previous frame.
    pub fn depacketize(&mut self, packet: &[u8]) -> Option<St2110JxsFrame> {
        if packet.len() < RTP_HEADER_LEN + JXS_HEADER_LEN || packet[0] & 0xC0 != 0x80 {
            return None;
        }
        let marker = packet[1] & 0x80 != 0;
        let rtp_timestamp = u32::from_be_bytes([packet[4], packet[5], packet[6], packet[7]]);

        // RFC 9134 payload header: reject slice mode (K=1), which we do not implement.
        let hdr = &packet[RTP_HEADER_LEN..RTP_HEADER_LEN + JXS_HEADER_LEN];
        let k = hdr[0] & 0x40 != 0;
        if k {
            return None;
        }
        let payload = &packet[RTP_HEADER_LEN + JXS_HEADER_LEN..];

        // A new frame's timestamp before the previous frame's marker means the marker
        // packet was lost: drop the stale partial rather than run two frames together.
        match self.ts {
            Some(t) if t != rtp_timestamp => {
                self.buf.clear();
                self.ts = Some(rtp_timestamp);
            }
            None => self.ts = Some(rtp_timestamp),
            _ => {}
        }

        // Bound the reassembly buffer; a run-on frame is discarded, not accumulated.
        if self.buf.len().saturating_add(payload.len()) > self.max_frame_bytes {
            self.buf.clear();
            self.ts = None;
            return None;
        }
        self.buf.extend_from_slice(payload);

        if marker {
            let codestream = core::mem::take(&mut self.buf);
            self.ts = None;
            Some(St2110JxsFrame {
                rtp_timestamp,
                codestream,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// A deterministic pseudo-codestream of `len` distinct bytes.
    fn codestream(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 31 + 7) as u8).collect()
    }

    #[test]
    fn round_trips_across_multiple_packets() {
        // A small MTU forces the codestream across several packets.
        let cs = codestream(1000);
        let tai = 1_700_000_000_000_000_000u64;
        let mut tx =
            St2110JxsPacketizer::new(96, 0x4A58_5300, RTP_HEADER_LEN + JXS_HEADER_LEN + 300);
        let packets = tx.packetize(&cs, tai);
        assert!(packets.len() > 1, "small MTU splits the codestream");

        // Only the last packet carries the marker; all share the 90 kHz timestamp.
        let want_ts = MediaClock::video()
            .rtp_timestamp(g2g_core::TaiNs(tai))
            .get();
        for (i, p) in packets.iter().enumerate() {
            assert_eq!(
                p[1] & 0x80 != 0,
                i + 1 == packets.len(),
                "marker only on last"
            );
            let ts = u32::from_be_bytes([p[4], p[5], p[6], p[7]]);
            assert_eq!(ts, want_ts, "every packet shares the frame timestamp");
            // Codestream mode: K bit clear, T bit set on every packet.
            assert_eq!(p[RTP_HEADER_LEN] & 0x40, 0, "K=0 codestream mode");
            assert_eq!(p[RTP_HEADER_LEN] & 0x80, 0x80, "T=1 sequential");
        }
        // The L bit is set on exactly the last packet.
        let last = packets.last().unwrap();
        assert_eq!(last[RTP_HEADER_LEN] & 0x20, 0x20, "L=1 on the final packet");

        let mut rx = St2110JxsDepacketizer::new(1 << 20);
        let mut done = None;
        for p in &packets {
            if let Some(f) = rx.depacketize(p) {
                done = Some(f);
            }
        }
        let f = done.expect("frame completes on the marker");
        assert_eq!(f.rtp_timestamp, want_ts);
        assert_eq!(f.codestream, cs, "the codestream survives the round trip");
    }

    #[test]
    fn single_packet_frame_and_frame_counter_advances() {
        let mut tx = St2110JxsPacketizer::new(112, 7, 1500);
        // Two consecutive frames; the F counter (bits in the payload header) advances.
        let p0 = tx.packetize(&codestream(50), 1_000_000_000);
        let p1 = tx.packetize(&codestream(60), 1_000_000_000 + 16_000_000);
        assert_eq!(p0.len(), 1);
        assert_eq!(p1.len(), 1);
        // The F counter is bits 26..22 of the 32-bit payload header word.
        let f = |pkt: &[u8]| {
            let h = &pkt[RTP_HEADER_LEN..RTP_HEADER_LEN + JXS_HEADER_LEN];
            (u32::from_be_bytes([h[0], h[1], h[2], h[3]]) >> 22) & 0x1F
        };
        assert_eq!(f(&p0[0]), 0);
        assert_eq!(f(&p1[0]), 1, "frame counter increments per frame");
    }

    #[test]
    fn drops_slice_mode_and_short_packets() {
        let mut rx = St2110JxsDepacketizer::new(1 << 20);
        assert!(
            rx.depacketize(&[0u8; 8]).is_none(),
            "shorter than RTP+JXS header"
        );
        // A well-formed RTP+JXS header but K=1 (slice mode) is dropped.
        let mut slice = vec![0x80, 0x80 | 96, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0];
        slice.extend_from_slice(&jxs_header(true, 0, 0)); // K=0...
        slice[RTP_HEADER_LEN] |= 0x40; // ...flip K to slice mode
        slice.extend_from_slice(&[1, 2, 3]);
        assert!(rx.depacketize(&slice).is_none(), "slice mode dropped");
    }

    #[test]
    fn bounds_a_runaway_frame() {
        // A sender that never sets the marker cannot grow the buffer past the cap.
        let mut tx = St2110JxsPacketizer::new(96, 1, RTP_HEADER_LEN + JXS_HEADER_LEN + 100);
        let packets = tx.packetize(&codestream(10_000), 1_000_000_000);
        let mut rx = St2110JxsDepacketizer::new(256); // far below the frame size
        let mut completed = false;
        for p in &packets {
            // Strip the marker so nothing ever completes: force unbounded growth.
            let mut p = p.clone();
            p[1] &= !0x80;
            if rx.depacketize(&p).is_some() {
                completed = true;
            }
        }
        assert!(!completed, "no frame completes without a marker");
        assert!(rx.buf.len() <= 256, "reassembly buffer stays bounded");
    }

    #[test]
    fn lost_marker_discards_the_stale_partial() {
        let mut tx = St2110JxsPacketizer::new(96, 1, RTP_HEADER_LEN + JXS_HEADER_LEN + 20);
        let frame_a = tx.packetize(&codestream(100), 1_000_000_000);
        let frame_b = tx.packetize(&codestream(30), 1_000_000_000 + 16_000_000);
        let mut rx = St2110JxsDepacketizer::new(1 << 20);
        // Feed frame A but drop its final (marker) packet.
        for p in &frame_a[..frame_a.len() - 1] {
            assert!(rx.depacketize(p).is_none());
        }
        // Frame B arrives (new timestamp) and completes; A's partial is discarded.
        let mut out = None;
        for p in &frame_b {
            if let Some(f) = rx.depacketize(p) {
                out = Some(f);
            }
        }
        let f = out.expect("frame B completes");
        assert_eq!(
            f.codestream,
            codestream(30),
            "only frame B, not A's leftovers"
        );
    }
}
