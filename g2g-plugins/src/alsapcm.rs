//! Shared libasound plumbing for the ALSA render sink and capture source: the
//! `Caps::Audio` <-> `snd_pcm` format mapping and the device channel-map
//! permutation. Both directions face the same two problems (which ALSA format a
//! pipeline PCM format opens a device with, and how the device's channel order
//! relates to ours), so the mapping lives here rather than once per element.
//!
//! ## Channel order
//!
//! Pipeline PCM is interleaved in the WAV / ffmpeg order
//! ([`ChannelLayout::default_for`]), but ALSA reports its own per-device map:
//! a 5.1 device is typically `FL FR RL RR FC LFE`, not our `FL FR FC LFE BL BR`.
//! [`device_permutation`] reads that map (only valid after `hw_params`) and
//! yields the reorder a render path applies; a capture path applies its
//! [`invert`].

use alloc::vec::Vec;

use alsa::pcm::{ChmapPosition, Format, PCM};

use g2g_core::{AudioFormat, Caps, ChannelLayout, ChannelPosition, G2gError};

use crate::audioconvert::sample_bytes;

/// The PCM sample formats an ALSA device is opened with, and the ALSA format
/// each maps to. `PcmS24Le` is our 3-byte packed layout, ALSA's `S24_3LE`
/// (not `S24LE`, which is 24 bits inside a 32-bit container).
pub(crate) const FORMATS: [(AudioFormat, Format); 5] = [
    (AudioFormat::PcmU8, Format::U8),
    (AudioFormat::PcmS16Le, Format::S16LE),
    (AudioFormat::PcmS24Le, Format::S243LE),
    (AudioFormat::PcmS32Le, Format::S32LE),
    (AudioFormat::PcmF32Le, Format::FloatLE),
];

/// Negotiated PCM device parameters passed to a worker as one unit (keeps the
/// worker signatures under clippy's argument cap).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PcmConfig {
    /// Pipeline sample format: drives the byte-level frame arithmetic.
    pub(crate) format: AudioFormat,
    /// The ALSA format the device is opened with.
    pub(crate) fmt: Format,
    pub(crate) channels: u32,
    pub(crate) rate: u32,
}

impl PcmConfig {
    /// Bytes in one interleaved sample frame.
    pub(crate) fn frame_bytes(&self) -> usize {
        sample_bytes(self.format) * self.channels as usize
    }
}

/// Negotiated PCM parameters. Compressed audio (AAC / Opus) is rejected
/// structurally, as in `WasapiSink`.
pub(crate) fn alsa_params(caps: &Caps) -> Result<PcmConfig, G2gError> {
    let Caps::Audio {
        format,
        channels,
        sample_rate,
    } = caps
    else {
        return Err(G2gError::CapsMismatch);
    };
    let (_, fmt) = FORMATS
        .iter()
        .find(|(f, _)| f == format)
        .ok_or(G2gError::CapsMismatch)?;
    Ok(PcmConfig {
        format: *format,
        fmt: *fmt,
        channels: u32::from(*channels),
        rate: *sample_rate,
    })
}

/// Our speaker position for an ALSA channel-map position. `None` for one
/// outside the [`ChannelLayout`] table (or an unpositioned channel), which
/// makes the caller fall back to passing the buffer straight through.
fn our_position(p: ChmapPosition) -> Option<ChannelPosition> {
    Some(match p {
        ChmapPosition::FL => ChannelPosition::Fl,
        ChmapPosition::FR => ChannelPosition::Fr,
        ChmapPosition::FC | ChmapPosition::Mono => ChannelPosition::Fc,
        ChmapPosition::LFE => ChannelPosition::Lfe,
        ChmapPosition::RL => ChannelPosition::Bl,
        ChmapPosition::RR => ChannelPosition::Br,
        ChmapPosition::SL => ChannelPosition::Sl,
        ChmapPosition::SR => ChannelPosition::Sr,
        ChmapPosition::RC => ChannelPosition::Bc,
        ChmapPosition::FLC => ChannelPosition::Flc,
        ChmapPosition::FRC => ChannelPosition::Frc,
        _ => return None,
    })
}

/// The permutation from our interleave order (the default layout for
/// `channels`) into the device's reported channel map: `perm[d]` is the source
/// channel that feeds device channel `d`. `None` when the two orders already
/// agree, the device reports no map, or a device position falls outside our
/// layout: the buffer then goes out unpermuted. Must be called after
/// `hw_params`, which is when ALSA fixes the map.
pub(crate) fn device_permutation(pcm: &PCM, channels: u32) -> Option<Vec<usize>> {
    let layout = ChannelLayout::default_for(u8::try_from(channels).ok()?)?;
    let positions: Vec<ChmapPosition> = (&pcm.get_chmap().ok()?).into();
    if positions.len() != channels as usize {
        return None;
    }
    let mut perm = Vec::with_capacity(positions.len());
    for p in positions {
        perm.push(layout.index_of(our_position(p)?)?);
    }
    if perm.iter().enumerate().all(|(d, s)| d == *s) {
        return None;
    }
    Some(perm)
}

/// The reverse of a [`device_permutation`]: `inv[s]` is the device channel that
/// carries our channel `s`, the direction a capture path needs.
pub(crate) fn invert(perm: Vec<usize>) -> Vec<usize> {
    let mut inv = alloc::vec![0usize; perm.len()];
    for (d, &s) in perm.iter().enumerate() {
        inv[s] = d;
    }
    inv
}

/// Reorder interleaved frames: output channel `i` of each frame takes source
/// channel `perm[i]`. A ragged trailing partial frame is dropped (it cannot be
/// written as a frame anyway).
pub(crate) fn permute(bytes: &[u8], perm: &[usize], sample: usize) -> Vec<u8> {
    let frame = sample * perm.len();
    let mut out = Vec::with_capacity(bytes.len());
    for f in bytes.chunks_exact(frame) {
        for &s in perm {
            out.extend_from_slice(&f[s * sample..][..sample]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(format: AudioFormat, channels: u8, rate: u32) -> Caps {
        Caps::Audio {
            format,
            channels,
            sample_rate: rate,
        }
    }

    #[test]
    fn alsa_params_maps_formats_and_rejects_compressed() {
        let p = alsa_params(&caps(AudioFormat::PcmS16Le, 2, 48_000)).unwrap();
        assert_eq!((p.fmt, p.channels, p.rate), (Format::S16LE, 2, 48_000));
        assert_eq!(p.frame_bytes(), 4);
        let p = alsa_params(&caps(AudioFormat::PcmF32Le, 1, 44_100)).unwrap();
        assert_eq!((p.fmt, p.channels, p.rate), (Format::FloatLE, 1, 44_100));
        // 24-bit is the 3-byte packed ALSA format, not the 32-bit container.
        let p = alsa_params(&caps(AudioFormat::PcmS24Le, 6, 48_000)).unwrap();
        assert_eq!(p.fmt, Format::S243LE);
        assert_eq!(p.frame_bytes(), 18);
        assert_eq!(
            alsa_params(&caps(AudioFormat::PcmS32Le, 2, 48_000))
                .unwrap()
                .fmt,
            Format::S32LE
        );
        assert_eq!(
            alsa_params(&caps(AudioFormat::PcmU8, 2, 8_000))
                .unwrap()
                .fmt,
            Format::U8
        );
        assert_eq!(
            alsa_params(&caps(AudioFormat::Aac, 2, 48_000)),
            Err(G2gError::CapsMismatch)
        );
        // a raw format we do not open a device with is rejected too.
        assert_eq!(
            alsa_params(&caps(AudioFormat::Mulaw, 1, 8_000)),
            Err(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn permute_reorders_frames_into_the_device_map() {
        // 5.1: our FL FR FC LFE BL BR -> the usual ALSA FL FR RL RR FC LFE.
        let perm = [0usize, 1, 4, 5, 2, 3];
        // one frame of s16 channel markers 0..5.
        let src: Vec<u8> = (0i16..6).flat_map(|c| c.to_le_bytes()).collect();
        let out = permute(&src, &perm, 2);
        let got: Vec<i16> = out
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(got, [0, 1, 4, 5, 2, 3]);
    }

    #[test]
    fn permute_handles_three_byte_samples_and_drops_a_ragged_tail() {
        // S24 is 3 bytes: a swap of a stereo frame moves 3 bytes at a time.
        let src = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let out = permute(&src, &[1, 0], 3);
        assert_eq!(out, [4, 5, 6, 1, 2, 3]);
    }

    #[test]
    fn invert_undoes_a_permutation() {
        let perm = alloc::vec![0usize, 1, 4, 5, 2, 3];
        let inv = invert(perm.clone());
        assert_eq!(inv, [0, 1, 4, 5, 2, 3]);
        // capture-then-render round trip lands back on the original order.
        let src: Vec<u8> = (0i16..6).flat_map(|c| c.to_le_bytes()).collect();
        let back = permute(&permute(&src, &inv, 2), &perm, 2);
        assert_eq!(back, src);
    }
}
