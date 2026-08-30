//! Endian and signedness conversion at the AIFF / AU container boundary.
//!
//! g2g's PCM `AudioFormat`s are little-endian (and 8-bit unsigned). AIFF and AU
//! store multi-byte samples big-endian and 8-bit samples signed, so the parser
//! swaps into the graph's native form and the muxer swaps back. Width-aligned
//! leftovers stay with the caller: a trailing partial sample is not converted.

use g2g_core::AudioFormat;

/// Formats AIFF and AU both carry. One table so the muxer, parser, and tests
/// cannot drift.
pub(crate) const CARRIED: [AudioFormat; 7] = [
    AudioFormat::PcmU8,
    AudioFormat::PcmS16Le,
    AudioFormat::PcmS24Le,
    AudioFormat::PcmS32Le,
    AudioFormat::PcmF32Le,
    AudioFormat::Mulaw,
    AudioFormat::Alaw,
];

/// Bytes per sample of a PCM / companded format these containers carry, `None`
/// for a coded format they do not describe.
pub(crate) fn sample_width(format: AudioFormat) -> Option<usize> {
    match format {
        AudioFormat::PcmU8 | AudioFormat::Mulaw | AudioFormat::Alaw => Some(1),
        AudioFormat::PcmS16Le => Some(2),
        AudioFormat::PcmS24Le => Some(3),
        AudioFormat::PcmS32Le | AudioFormat::PcmF32Le => Some(4),
        _ => None,
    }
}

/// One PCM stream: the graph format it carries and how its samples sit on the
/// wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PcmWire {
    pub format: AudioFormat,
    pub channels: u8,
    pub sample_rate: u32,
    /// Multi-byte samples are big-endian (classic AIFF, AIFC `NONE`, AU).
    pub big_endian: bool,
}

pub(crate) fn read_u16_be(data: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes(data.get(at..at + 2)?.try_into().ok()?))
}

pub(crate) fn read_u32_be(data: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(data.get(at..at + 4)?.try_into().ok()?))
}

pub(crate) fn read_fourcc(data: &[u8], at: usize) -> Option<[u8; 4]> {
    data.get(at..at + 4)?.try_into().ok()
}

/// Swap each `width`-byte sample in place. `width` 1 is a no-op. Bytes past the
/// last whole sample are left untouched.
pub(crate) fn swap_endian(data: &mut [u8], width: usize) {
    if width <= 1 {
        return;
    }
    for sample in data.chunks_exact_mut(width) {
        sample.reverse();
    }
}

/// Map signed 8-bit PCM (AIFF / AU) onto unsigned 8-bit PCM (`PcmU8`) or the
/// reverse: XOR with 0x80, which is addition of 128 in two's complement.
pub(crate) fn xor_sign8(data: &mut [u8]) {
    for b in data {
        *b ^= 0x80;
    }
}

/// Convert a buffer between the file's on-wire layout and the graph's LE / U8
/// form. `big_endian` is whether multi-byte samples are big-endian on the wire.
/// 8-bit linear PCM is signed on the wire exactly when the graph format is
/// `PcmU8`; G.711 bytes are the same either way. This is its own inverse, so
/// parser and muxer both call it.
pub(crate) fn convert_wire_layout(data: &mut [u8], format: AudioFormat, big_endian: bool) {
    match sample_width(format) {
        Some(_) if format == AudioFormat::PcmU8 => xor_sign8(data),
        Some(width) if big_endian => swap_endian(data, width),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn every_carried_format_has_a_width() {
        for format in CARRIED {
            assert!(sample_width(format).is_some(), "{format:?}");
        }
    }

    #[test]
    fn endian_swap_is_an_involution() {
        for width in 2..=4 {
            let original: Vec<u8> = (0u8..width * 5).collect();
            let mut data = original.clone();
            swap_endian(&mut data, width as usize);
            assert_ne!(data, original, "width {width} must actually swap");
            swap_endian(&mut data, width as usize);
            assert_eq!(data, original, "width {width}");
        }
    }

    #[test]
    fn sign8_xor_is_an_involution() {
        let original: Vec<u8> = (0..=u8::MAX).collect();
        let mut data = original.clone();
        xor_sign8(&mut data);
        assert_ne!(data, original);
        xor_sign8(&mut data);
        assert_eq!(data, original);
    }

    #[test]
    fn wire_conversion_round_trips_every_carried_format() {
        for format in CARRIED {
            for big_endian in [false, true] {
                let original: Vec<u8> = (0u8..sample_width(format).unwrap() as u8 * 6).collect();
                let mut data = original.clone();
                convert_wire_layout(&mut data, format, big_endian);
                convert_wire_layout(&mut data, format, big_endian);
                assert_eq!(data, original, "{format:?}");
            }
        }
    }
}
