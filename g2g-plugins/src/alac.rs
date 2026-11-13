// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#[cfg(feature = "std")]
use g2g_core::G2gError;

// the `alac` atom: box header, version and flags, then the 24-byte ALACSpecificConfig
const ALAC_ATOM_LEN: usize = 36;

pub(crate) fn is_alac_atom(packet: &[u8]) -> bool {
    packet.len() == ALAC_ATOM_LEN && &packet[4..8] == b"alac"
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct AlacConfig {
    pub(crate) channels: u8,
    pub(crate) bit_depth: u8,
    pub(crate) sample_rate: u32,
}

#[cfg(feature = "std")]
pub(crate) fn alac_config(atom: &[u8]) -> Result<AlacConfig, G2gError> {
    const ALAC_ATOM_BIT_DEPTH_AT: usize = 17;
    const ALAC_ATOM_CHANNELS_AT: usize = 21;
    const ALAC_ATOM_SAMPLE_RATE_AT: usize = 32;
    if !is_alac_atom(atom) {
        return Err(G2gError::CapsMismatch);
    }
    Ok(AlacConfig {
        channels: atom[ALAC_ATOM_CHANNELS_AT],
        bit_depth: atom[ALAC_ATOM_BIT_DEPTH_AT],
        sample_rate: crate::mp4box::be32(atom, ALAC_ATOM_SAMPLE_RATE_AT)?,
    })
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn alac_config_reads_the_cookie_fields() {
        // ffmpeg's alac encoder wrote this atom for 24-bit 5.1 at 48 kHz
        let config = [
            0x00, 0x00, 0x10, 0x00, 0x00, 0x18, 0x28, 0x0A, 0x0E, 0x06, 0x00, 0x00, 0x00, 0x01,
            0x20, 0x04, 0x00, 0x69, 0x78, 0x00, 0x00, 0x00, 0xBB, 0x80,
        ];
        let atom = crate::mp4box::mp4_box(b"alac", &[&[0u8; 4][..], &config].concat());
        let alac = alac_config(&atom).expect("an alac atom");
        assert_eq!(alac.channels, 6);
        assert_eq!(alac.bit_depth, 24);
        assert_eq!(alac.sample_rate, 48_000);
    }
}
