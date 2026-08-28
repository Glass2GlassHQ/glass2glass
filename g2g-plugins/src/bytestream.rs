//! What the raw byte-stream carriers (TCP, SRT, raw UDP) share: the frame shape
//! a received chunk takes, the MPEG-TS packet geometry that decides how a
//! datagram is cut, and the container list a wire sink can carry.

use alloc::vec::Vec;

use g2g_core::{ByteStreamEncoding, Caps};

/// MPEG-TS packet size (ISO/IEC 13818-1).
pub const TS_PACKET_SIZE: usize = 188;
/// TS packets per datagram in the 1316-byte payload SRT and the broadcast UDP
/// senders use: the largest whole number that fits under a 1500-byte MTU.
pub const TS_PACKETS_PER_DATAGRAM: usize = 7;
/// That payload, 1316 bytes.
pub const TS_DATAGRAM_PAYLOAD: usize = TS_PACKET_SIZE * TS_PACKETS_PER_DATAGRAM;

/// Formats a raw wire sink carries. `ByteStreamEncoding` is `#[non_exhaustive]`
/// from another crate, so this is the sink's own enumeration of it: a variant
/// added later is simply not advertised until it is listed here.
pub const CARRIED_ENCODINGS: [ByteStreamEncoding; 15] = [
    ByteStreamEncoding::MpegTs,
    ByteStreamEncoding::Matroska,
    ByteStreamEncoding::Ogg,
    ByteStreamEncoding::Flv,
    ByteStreamEncoding::IsoBmff,
    ByteStreamEncoding::Mp4,
    ByteStreamEncoding::Ivf,
    ByteStreamEncoding::MpegPs,
    ByteStreamEncoding::Wav,
    ByteStreamEncoding::Avi,
    ByteStreamEncoding::Rtp,
    ByteStreamEncoding::Srtp,
    ByteStreamEncoding::Rtcp,
    ByteStreamEncoding::Srtcp,
    ByteStreamEncoding::Dtls,
];

/// Whether each frame is one complete network packet and must remain one
/// datagram.
pub fn is_packet_encoding(encoding: ByteStreamEncoding) -> bool {
    matches!(
        encoding,
        ByteStreamEncoding::Rtp
            | ByteStreamEncoding::Srtp
            | ByteStreamEncoding::Rtcp
            | ByteStreamEncoding::Srtcp
            | ByteStreamEncoding::Dtls
    )
}

/// [`CARRIED_ENCODINGS`] as caps, for a pad template or an `Accepts` set.
pub fn carried_bytestream_caps() -> Vec<Caps> {
    CARRIED_ENCODINGS
        .iter()
        .map(|&encoding| Caps::ByteStream { encoding })
        .collect()
}

/// Bytes per datagram for `encoding` under a `max_payload` cap. An MPEG-TS
/// stream is cut on whole 188-byte packets, so a receiver never has to
/// reassemble a split one; with the 1400-byte default that is
/// [`TS_DATAGRAM_PAYLOAD`].
pub fn datagram_chunk(encoding: ByteStreamEncoding, max_payload: usize) -> usize {
    if encoding != ByteStreamEncoding::MpegTs {
        return max_payload.max(1);
    }
    (max_payload / TS_PACKET_SIZE).max(1) * TS_PACKET_SIZE
}

/// A frame carrying `bytes`, stamped and sequenced the way `FileSrc` stamps a
/// file chunk, so every byte source looks identical downstream.
#[cfg(any(feature = "tcp", feature = "udp-ingress"))]
pub(crate) fn byte_frame(bytes: Vec<u8>, sequence: u64) -> g2g_core::frame::Frame {
    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{FrameTiming, MemoryDomain};

    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            arrival_ns: g2g_core::metrics::monotonic_ns(),
            ..FrameTiming::default()
        },
        sequence,
        meta: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ts_datagrams_hold_whole_packets() {
        // The 1400-byte udpsink default has to land on the SRT payload size.
        assert_eq!(
            datagram_chunk(ByteStreamEncoding::MpegTs, 1400),
            TS_DATAGRAM_PAYLOAD
        );
        assert_eq!(
            datagram_chunk(ByteStreamEncoding::MpegTs, 1400) % TS_PACKET_SIZE,
            0
        );
        // Below one packet the chunk still carries a whole packet: a split TS
        // packet is unreadable to the receiver either way.
        assert_eq!(
            datagram_chunk(ByteStreamEncoding::MpegTs, 100),
            TS_PACKET_SIZE
        );
    }

    #[test]
    fn other_containers_use_the_whole_payload() {
        assert_eq!(datagram_chunk(ByteStreamEncoding::Matroska, 1400), 1400);
    }

    #[test]
    fn packet_formats_are_carried_without_container_conversion() {
        for encoding in [
            ByteStreamEncoding::Rtp,
            ByteStreamEncoding::Srtp,
            ByteStreamEncoding::Rtcp,
            ByteStreamEncoding::Srtcp,
            ByteStreamEncoding::Dtls,
        ] {
            assert!(CARRIED_ENCODINGS.contains(&encoding));
            assert!(is_packet_encoding(encoding));
        }
    }
}
