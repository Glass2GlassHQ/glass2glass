//! ST 2110-30 PCM audio over RTP (M595): the packetizer and depacketizer for
//! uncompressed linear PCM, AES67-compatible (RFC 3190 L16 / L24).
//!
//! ST 2110-30 carries interleaved big-endian PCM straight in the RTP payload (no
//! per-packet media header, unlike -20 video or -40 ancillary), with the RTP
//! timestamp taken from the [`MediaClock`] at the PTP sampling instant of the
//! packet's first sample. The timestamp then advances by the sample-frame count
//! per packet, so a receiver locked to the same grandmaster reconstructs the
//! exact playout time, the audio half of A/V sync across devices.
//!
//! This is the sans-IO core (samples <-> RTP packets), like `rtppay`: pure
//! `no_std` + alloc, so it is CI round-trip testable and an element wrapper can
//! sit on top later. Samples are carried as `i32` (L16 uses the low 16 bits, L24
//! the low 24), matching how a pipeline holds 16- and 24-bit PCM.

use alloc::vec::Vec;

use g2g_core::rtp::{RtpHeader, RTP_HEADER_LEN};
use g2g_core::MediaClock;


/// PCM sample depth on the wire (ST 2110-30 permits 16- and 24-bit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleDepth {
    /// 16-bit big-endian (RFC 3190 L16).
    L16,
    /// 24-bit big-endian (RFC 3190 L24), the common professional depth.
    L24,
}

impl SampleDepth {
    /// Bytes per sample on the wire.
    pub fn bytes(self) -> usize {
        match self {
            Self::L16 => 2,
            Self::L24 => 3,
        }
    }

    /// Write one interleaved sample (low `bytes()` bits of `v`) big-endian.
    fn write(self, v: i32, out: &mut Vec<u8>) {
        match self {
            Self::L16 => out.extend_from_slice(&(v as i16).to_be_bytes()),
            Self::L24 => out.extend_from_slice(&[(v >> 16) as u8, (v >> 8) as u8, v as u8]),
        }
    }

    /// Read one big-endian sample, sign-extended to `i32`.
    fn read(self, b: &[u8]) -> i32 {
        match self {
            Self::L16 => i32::from(i16::from_be_bytes([b[0], b[1]])),
            Self::L24 => {
                let raw = (i32::from(b[0]) << 16) | (i32::from(b[1]) << 8) | i32::from(b[2]);
                // Sign-extend the 24-bit value.
                (raw << 8) >> 8
            }
        }
    }
}

/// Packetizes interleaved PCM into ST 2110-30 RTP packets.
#[derive(Debug)]
pub struct St2110AudioPacketizer {
    payload_type: u8,
    ssrc: u32,
    sequence: u16,
    channels: u16,
    depth: SampleDepth,
    clock: MediaClock,
    frames_per_packet: usize,
}

impl St2110AudioPacketizer {
    /// A packetizer for `channels` at `sample_rate_hz`, emitting `frames_per_packet`
    /// sample-frames per RTP packet (e.g. 48 for 1 ms at 48 kHz, 6 for 125 us).
    /// `payload_type` is the dynamic RTP PT; `ssrc` the stream source id.
    pub fn new(
        payload_type: u8,
        ssrc: u32,
        sample_rate_hz: u32,
        channels: u16,
        depth: SampleDepth,
        frames_per_packet: usize,
    ) -> Self {
        Self {
            payload_type: payload_type & 0x7f,
            ssrc,
            sequence: 0,
            channels: channels.max(1),
            depth,
            clock: MediaClock::audio(sample_rate_hz),
            frames_per_packet: frames_per_packet.max(1),
        }
    }

    /// The media clock (for recovering a packet's PTP time on the receive side).
    pub fn media_clock(&self) -> MediaClock {
        self.clock
    }

    /// Packetize interleaved PCM (`channels` samples per frame). `first_sample_tai_ns`
    /// is the PTP/TAI sampling time of the first frame; each packet's RTP timestamp
    /// is that instant plus its sample-frame offset. A trailing partial frame is
    /// ignored (never fragment a sample-frame across packets).
    pub fn packetize(&mut self, samples: &[i32], first_sample_tai_ns: u64) -> Vec<Vec<u8>> {
        let ch = usize::from(self.channels);
        let total_frames = samples.len() / ch;
        // RTP timestamps advance by exact sample counts from the first frame's
        // media-clock value, not by re-deriving from ns each packet.
        let base_ts = self.clock.rtp_timestamp(g2g_core::TaiNs(first_sample_tai_ns)).get();

        let mut packets = Vec::new();
        let mut frame = 0usize;
        while frame < total_frames {
            let frames = self.frames_per_packet.min(total_frames - frame);
            let rtp_ts = base_ts.wrapping_add(frame as u32);

            let mut pkt = Vec::with_capacity(RTP_HEADER_LEN + frames * ch * self.depth.bytes());
            // Audio is a continuous stream: the marker bit stays clear.
            self.write_header(&mut pkt, rtp_ts);

            let start = frame * ch;
            let end = start + frames * ch;
            for &s in &samples[start..end] {
                self.depth.write(s, &mut pkt);
            }
            packets.push(pkt);

            self.sequence = self.sequence.wrapping_add(1);
            frame += frames;
        }
        packets
    }

    fn write_header(&self, out: &mut Vec<u8>, timestamp: u32) {
        let header = RtpHeader {
            payload_type: self.payload_type,
            marker: false,
            sequence: self.sequence,
            timestamp,
            ssrc: self.ssrc,
        };
        out.extend_from_slice(&header.to_bytes());
    }
}

/// A depacketized ST 2110-30 packet: its RTP fields plus interleaved samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct St2110AudioPacket {
    pub sequence: u16,
    pub rtp_timestamp: u32,
    /// Interleaved PCM (`channels` per frame), sign-extended to `i32`.
    pub samples: Vec<i32>,
}

/// Depacketizes ST 2110-30 RTP packets back to interleaved PCM.
#[derive(Debug)]
pub struct St2110AudioDepacketizer {
    channels: u16,
    depth: SampleDepth,
}

impl St2110AudioDepacketizer {
    /// A depacketizer for `channels` at `depth` (must match the sender).
    pub fn new(channels: u16, depth: SampleDepth) -> Self {
        Self { channels: channels.max(1), depth }
    }

    /// Parse one RTP packet into its sequence, timestamp, and interleaved samples,
    /// or `None` if it is too short or the payload is not a whole number of
    /// sample-frames.
    pub fn depacketize(&self, packet: &[u8]) -> Option<St2110AudioPacket> {
        if packet.len() < RTP_HEADER_LEN {
            return None;
        }
        // Only version 2 with no CSRC list / extension (what we send / -30 uses).
        if packet[0] & 0xc0 != 0x80 {
            return None;
        }
        let sequence = u16::from_be_bytes([packet[2], packet[3]]);
        let rtp_timestamp = u32::from_be_bytes([packet[4], packet[5], packet[6], packet[7]]);

        let payload = &packet[RTP_HEADER_LEN..];
        let sample_bytes = self.depth.bytes();
        let frame_bytes = sample_bytes * usize::from(self.channels);
        // A partial sample-frame is malformed; reject rather than mis-deinterleave.
        if frame_bytes == 0 || payload.len() % frame_bytes != 0 {
            return None;
        }
        let mut samples = Vec::with_capacity(payload.len() / sample_bytes);
        for chunk in payload.chunks_exact(sample_bytes) {
            samples.push(self.depth.read(chunk));
        }
        Some(St2110AudioPacket { sequence, rtp_timestamp, samples })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// 48 kHz, so 1 ms = 48 sample-frames per packet.
    const RATE: u32 = 48_000;
    const FRAMES_PER_PKT: usize = 48;

    /// Interleaved stereo PCM: `frames` frames, distinct L/R values incl negatives.
    fn stereo(frames: usize) -> Vec<i32> {
        let mut s = Vec::with_capacity(frames * 2);
        for i in 0..frames as i32 {
            s.push(i * 137 - 3000); // left, crosses zero
            s.push(-(i * 91) - 1); // right, always negative
        }
        s
    }

    #[test]
    fn round_trips_l24_stereo_across_packets() {
        let tai = 1_700_000_000_000_000_000u64;
        let mut tx = St2110AudioPacketizer::new(97, 0xCAFE, RATE, 2, SampleDepth::L24, FRAMES_PER_PKT);
        let rx = St2110AudioDepacketizer::new(2, SampleDepth::L24);
        let input = stereo(120); // 120 frames -> 3 packets (48 + 48 + 24)

        let packets = tx.packetize(&input, tai);
        assert_eq!(packets.len(), 3, "120 frames at 48/pkt -> 3 packets");

        // Timestamps advance by the sample-frame count; the first equals the
        // media clock at the sampling instant.
        let clock = MediaClock::audio(RATE);
        let mut out = Vec::new();
        for (i, pkt) in packets.iter().enumerate() {
            let p = rx.depacketize(pkt).expect("valid packet");
            assert_eq!(p.sequence, i as u16, "sequence increments per packet");
            assert_eq!(p.rtp_timestamp, clock.rtp_timestamp(g2g_core::TaiNs(tai)).wrapping_add((i * FRAMES_PER_PKT) as u32).get());
            out.extend_from_slice(&p.samples);
        }
        assert_eq!(out, input, "L24 PCM survives the round trip, sign-extended");
    }

    #[test]
    fn round_trips_l16() {
        let mut tx = St2110AudioPacketizer::new(96, 1, RATE, 2, SampleDepth::L16, FRAMES_PER_PKT);
        let rx = St2110AudioDepacketizer::new(2, SampleDepth::L16);
        // Values within i16 range (incl the extremes) for L16.
        let input: Vec<i32> = [-32768, 32767, 0, -1, 12345, -6000].to_vec();
        let packets = tx.packetize(&input, 1_700_000_000_000_000_000);
        assert_eq!(packets.len(), 1);
        let p = rx.depacketize(&packets[0]).unwrap();
        assert_eq!(p.samples, input, "L16 PCM incl full-scale round-trips");
    }

    #[test]
    fn timestamp_is_the_media_clock_at_the_sampling_instant() {
        let tai = 1_700_000_123_456_789_000u64;
        let mut tx = St2110AudioPacketizer::new(96, 7, RATE, 1, SampleDepth::L16, FRAMES_PER_PKT);
        let packets = tx.packetize(&[0; FRAMES_PER_PKT], tai);
        let rx = St2110AudioDepacketizer::new(1, SampleDepth::L16);
        let p = rx.depacketize(&packets[0]).unwrap();
        // A receiver recovers the sampling time from the timestamp + its PTP clock.
        let recovered = MediaClock::audio(RATE)
            .tai_from_rtp(g2g_core::RtpTs(p.rtp_timestamp), g2g_core::TaiNs(tai + 5_000_000))
            .get();
        assert!(tai.abs_diff(recovered) <= MediaClock::audio(RATE).ticks_to_ns(1) + 1);
    }

    #[test]
    fn rejects_short_and_ragged_packets() {
        let rx = St2110AudioDepacketizer::new(2, SampleDepth::L24);
        assert!(rx.depacketize(&[0u8; 8]).is_none(), "shorter than an RTP header");
        // A header plus a payload that is not a whole stereo L24 frame (6 bytes).
        let mut ragged = vec![0x80, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        ragged.extend_from_slice(&[1, 2, 3, 4]); // 4 bytes, not a multiple of 6
        assert!(rx.depacketize(&ragged).is_none(), "partial sample-frame rejected");
    }
}
