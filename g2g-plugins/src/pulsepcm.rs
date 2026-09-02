//! Shared libpulse plumbing for the PulseAudio render sink and capture source:
//! the `Caps::Audio` <-> `pa_sample_spec` mapping and the channel map built from
//! our interleave order. Both directions want the same explicit map, so the
//! server routes / picks up channels by speaker instead of guessing positionally.

use libpulse_binding::channelmap::{Map, Position};
use libpulse_binding::sample::{Format as PaFormat, Spec};

use g2g_core::{AudioFormat, Caps, ChannelLayout, ChannelPosition, G2gError};

/// The PCM sample formats a PulseAudio stream is opened with, and the pulse
/// format each maps to. `PcmS24Le` is our 3-byte packed layout, pulse's packed
/// `S24le` (not `S24_32le`, which pads to 32 bits).
pub(crate) const FORMATS: [(AudioFormat, PaFormat); 5] = [
    (AudioFormat::PcmU8, PaFormat::U8),
    (AudioFormat::PcmS16Le, PaFormat::S16le),
    (AudioFormat::PcmS24Le, PaFormat::S24le),
    (AudioFormat::PcmS32Le, PaFormat::S32le),
    (AudioFormat::PcmF32Le, PaFormat::F32le),
];

/// Negotiated PCM parameters as a PulseAudio `Spec`. Compressed audio
/// (AAC / Opus) is rejected structurally, as in `WasapiSink`.
pub(crate) fn pulse_spec(caps: &Caps) -> Result<Spec, G2gError> {
    let Caps::Audio {
        format,
        channels,
        sample_rate,
        ..
    } = caps
    else {
        return Err(G2gError::CapsMismatch);
    };
    let (_, format) = FORMATS
        .iter()
        .find(|(f, _)| f == format)
        .ok_or(G2gError::CapsMismatch)?;
    let spec = Spec {
        format: *format,
        channels: *channels,
        rate: *sample_rate,
    };
    if !spec.is_valid() {
        return Err(G2gError::CapsMismatch);
    }
    Ok(spec)
}

fn pulse_position(p: ChannelPosition) -> Position {
    match p {
        ChannelPosition::Fl => Position::FrontLeft,
        ChannelPosition::Fr => Position::FrontRight,
        ChannelPosition::Fc => Position::FrontCenter,
        ChannelPosition::Lfe => Position::Lfe,
        ChannelPosition::Bl => Position::RearLeft,
        ChannelPosition::Br => Position::RearRight,
        ChannelPosition::Flc => Position::FrontLeftOfCenter,
        ChannelPosition::Frc => Position::FrontRightOfCenter,
        ChannelPosition::Bc => Position::RearCenter,
        ChannelPosition::Sl => Position::SideLeft,
        ChannelPosition::Sr => Position::SideRight,
    }
}

/// The channel map for our interleave order, so the server routes by speaker.
/// `None` past the layout table (> 8 channels), where the stream falls back to
/// the server's default map.
pub(crate) fn pulse_map(channels: u8) -> Option<Map> {
    let mut map = Map::default();
    // pulse spells a lone channel `mono`, not front-center.
    if channels == 1 {
        map.init_mono();
        return Some(map);
    }
    let layout = ChannelLayout::default_for(channels)?;
    map.init();
    map.set_len(channels);
    for (slot, pos) in map.get_mut().iter_mut().zip(layout.positions()) {
        *slot = pulse_position(pos);
    }
    map.is_valid().then_some(map)
}

/// An empty property string means "let the server choose" (the default server /
/// default device), which the libpulse API spells as `None`.
/// Only `pulsesrc` exposes server/device properties; `pulsesink` always opens
/// the defaults.
#[cfg(feature = "pulse-src")]
pub(crate) fn opt_name(s: &str) -> Option<&str> {
    (!s.is_empty()).then_some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pulse_spec_maps_formats_and_rejects_compressed() {
        let s16 = pulse_spec(&Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .expect("s16 spec");
        assert_eq!(s16.format, PaFormat::S16le);
        assert_eq!((s16.channels, s16.rate), (2, 48_000));

        let f32 = pulse_spec(&Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 1,
            sample_rate: 44_100,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .expect("f32 spec");
        assert_eq!(f32.format, PaFormat::F32le);

        let aac = Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(pulse_spec(&aac), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn pulse_spec_maps_the_wide_sample_formats() {
        let spec = |format, channels| {
            pulse_spec(&Caps::Audio {
                format,
                channels,
                sample_rate: 48_000,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            })
        };
        // 24-bit maps to the packed pulse format, not the 32-bit-padded one.
        assert_eq!(
            spec(AudioFormat::PcmS24Le, 6).unwrap().format,
            PaFormat::S24le
        );
        assert_eq!(
            spec(AudioFormat::PcmS32Le, 2).unwrap().format,
            PaFormat::S32le
        );
        assert_eq!(spec(AudioFormat::PcmU8, 2).unwrap().format, PaFormat::U8);
        // a raw format we do not open a stream with is rejected.
        assert_eq!(spec(AudioFormat::Alaw, 1), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn channel_map_follows_our_interleave_order() {
        let map = pulse_map(6).expect("5.1 has a layout");
        assert_eq!(
            map.get(),
            [
                Position::FrontLeft,
                Position::FrontRight,
                Position::FrontCenter,
                Position::Lfe,
                Position::RearLeft,
                Position::RearRight,
            ]
        );
        assert!(map.is_valid());
        let map = pulse_map(8).expect("7.1 has a layout");
        assert_eq!(map.get()[6..], [Position::SideLeft, Position::SideRight]);
        assert_eq!(pulse_map(1).unwrap().get(), [Position::Mono]);
        // past the layout table the server's default map applies.
        assert!(pulse_map(9).is_none());
    }

    #[cfg(feature = "pulse-src")]
    #[test]
    fn empty_names_mean_server_default() {
        assert_eq!(opt_name(""), None);
        assert_eq!(opt_name("alsa_input.pci-0000"), Some("alsa_input.pci-0000"));
    }
}
