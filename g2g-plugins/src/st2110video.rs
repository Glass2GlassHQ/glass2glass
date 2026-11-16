// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec::Vec;

use g2g_core::rtp::RTP_HEADER_LEN;

/// `packet` as a real sender may shape it: one CSRC, a one-word RFC 8285
/// extension and four bytes of padding around the same payload.
pub(crate) fn with_csrc_extension_and_padding(packet: &[u8]) -> Vec<u8> {
    const PADDING_FLAG: u8 = 0x20;
    const EXTENSION_FLAG: u8 = 0x10;
    const ONE_CSRC: u8 = 1;
    const CSRC: [u8; 4] = [0xC5, 0xC5, 0xC5, 0xC5];
    const EXTENSION: [u8; 8] = [0xBE, 0xDE, 0x00, 0x01, 0x10, 0xAA, 0x00, 0x00];
    const PADDING: [u8; 4] = [0, 0, 0, 4];
    let mut out = packet[..RTP_HEADER_LEN].to_vec();
    out[0] |= PADDING_FLAG | EXTENSION_FLAG | ONE_CSRC;
    out.extend_from_slice(&CSRC);
    out.extend_from_slice(&EXTENSION);
    out.extend_from_slice(&packet[RTP_HEADER_LEN..]);
    out.extend_from_slice(&PADDING);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::st2110video::{Sampling, St2110VideoDepacketizer, St2110VideoPacketizer};
    use g2g_core::RawVideoFormat;

    /// A deterministic packed frame of `stride*height` bytes with distinct values.
    fn ramp(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 7 + 3) as u8).collect()
    }

    #[test]
    fn depacketizes_past_csrcs_extension_and_padding() {
        let (w, h) = (8usize, 4usize);
        let frame = ramp(w * 4 * h);
        let mut tx = St2110VideoPacketizer::new(96, 0xF00D, Sampling::Rgba8, 1400);
        let packets = tx.packetize(&frame, w, h, 0).expect("packetizes");
        let mut rx = St2110VideoDepacketizer::new(RawVideoFormat::Rgba8, w, h).unwrap();
        let done = packets
            .iter()
            .filter_map(|p| rx.depacketize(&with_csrc_extension_and_padding(p)))
            .last();
        assert_eq!(done.expect("frame completes").bytes, frame);
    }
}
