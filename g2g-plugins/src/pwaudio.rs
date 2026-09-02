//! Shared PipeWire audio helpers for [`PipeWireSink`](crate::pipewiresink) and
//! [`PipeWireSrc`](crate::pipewiresrc): map our PCM `Caps` to an SPA audio
//! format pod for `Stream::connect`. Linux-only (`pipewire` feature).

use alloc::vec::Vec;

use pipewire::spa;
use spa::param::audio::{AudioFormat, AudioInfoRaw, MAX_CHANNELS};
use spa::param::ParamType;
use spa::pod::serialize::PodSerializer;
use spa::pod::{Object, Value};
use spa::utils::SpaTypes;

use g2g_core::{AudioFormat as G2gAudioFormat, Caps, ChannelLayout, ChannelPosition, G2gError};

use crate::audioconvert::sample_bytes;

/// The PCM sample formats the PipeWire elements open a stream with, and the SPA
/// format each maps to. `PcmS24Le` is our 3-byte packed layout, SPA's `S24LE`
/// (`S24_32LE` is the 32-bit-container variant).
const FORMATS: [(G2gAudioFormat, AudioFormat); 5] = [
    (G2gAudioFormat::PcmU8, AudioFormat::U8),
    (G2gAudioFormat::PcmS16Le, AudioFormat::S16LE),
    (G2gAudioFormat::PcmS24Le, AudioFormat::S24LE),
    (G2gAudioFormat::PcmS32Le, AudioFormat::S32LE),
    (G2gAudioFormat::PcmF32Le, AudioFormat::F32LE),
];

/// PCM parameters of an accepted `Caps::Audio`: (SPA format, channels, rate).
/// Compressed audio (AAC / Opus) is rejected structurally.
pub(crate) fn pw_params(caps: &Caps) -> Result<(AudioFormat, u32, u32), G2gError> {
    let Caps::Audio {
        format,
        channels,
        sample_rate,
        ..
    } = caps
    else {
        return Err(G2gError::CapsMismatch);
    };
    let (_, fmt) = FORMATS
        .iter()
        .find(|(f, _)| f == format)
        .ok_or(G2gError::CapsMismatch)?;
    Ok((*fmt, u32::from(*channels), *sample_rate))
}

/// Bytes per interleaved sample frame for an SPA audio format.
pub(crate) fn frame_bytes(format: AudioFormat, channels: u32) -> usize {
    FORMATS
        .iter()
        .find(|(_, spa)| *spa == format)
        .map_or(0, |(g2g, _)| sample_bytes(*g2g))
        * channels as usize
}

/// The SPA channel id for one of our speaker positions.
fn spa_position(p: ChannelPosition) -> u32 {
    match p {
        ChannelPosition::Fl => spa::sys::SPA_AUDIO_CHANNEL_FL,
        ChannelPosition::Fr => spa::sys::SPA_AUDIO_CHANNEL_FR,
        ChannelPosition::Fc => spa::sys::SPA_AUDIO_CHANNEL_FC,
        ChannelPosition::Lfe => spa::sys::SPA_AUDIO_CHANNEL_LFE,
        ChannelPosition::Bl => spa::sys::SPA_AUDIO_CHANNEL_RL,
        ChannelPosition::Br => spa::sys::SPA_AUDIO_CHANNEL_RR,
        ChannelPosition::Flc => spa::sys::SPA_AUDIO_CHANNEL_FLC,
        ChannelPosition::Frc => spa::sys::SPA_AUDIO_CHANNEL_FRC,
        ChannelPosition::Bc => spa::sys::SPA_AUDIO_CHANNEL_RC,
        ChannelPosition::Sl => spa::sys::SPA_AUDIO_CHANNEL_SL,
        ChannelPosition::Sr => spa::sys::SPA_AUDIO_CHANNEL_SR,
    }
}

/// The SPA position array for our interleave order (the default layout for
/// `channels`), so PipeWire routes a > 2-channel stream by speaker instead of
/// treating it as unpositioned. `None` past the layout table (> 8 channels).
fn spa_positions(channels: u32) -> Option<[u32; MAX_CHANNELS]> {
    let mut position = [0u32; MAX_CHANNELS];
    // pipewire spells a lone channel `mono`, not front-center.
    if channels == 1 {
        position[0] = spa::sys::SPA_AUDIO_CHANNEL_MONO;
        return Some(position);
    }
    let layout = ChannelLayout::default_for(u8::try_from(channels).ok()?)?;
    for (slot, p) in position.iter_mut().zip(layout.positions()) {
        *slot = spa_position(p);
    }
    Some(position)
}

/// Serialize a fixed `EnumFormat` audio pod (one value) for `Stream::connect`.
/// The returned bytes back a `Pod::from_bytes` at the call site (kept there so
/// the borrow lives as long as the `connect` call needs it).
pub(crate) fn format_pod_bytes(format: AudioFormat, channels: u32, rate: u32) -> Vec<u8> {
    let mut info = AudioInfoRaw::new();
    info.set_format(format);
    info.set_rate(rate);
    info.set_channels(channels);
    if let Some(position) = spa_positions(channels) {
        info.set_position(position);
    }
    let obj = Object {
        type_: SpaTypes::ObjectParamFormat.as_raw(),
        id: ParamType::EnumFormat.as_raw(),
        properties: info.into(),
    };
    PodSerializer::serialize(std::io::Cursor::new(Vec::new()), &Value::Object(obj))
        .expect("serialize SPA audio format pod")
        .0
        .into_inner()
}

/// The stream's current `pw_time`, `None` when the probe fails. Call only
/// while `stream` is live (inside its own process callback, or with the
/// loop locked): the raw pointer is read without a lifetime tie.
pub(crate) fn stream_time(stream: &pipewire::stream::StreamRef) -> Option<pipewire::sys::pw_time> {
    // SAFETY: zero is a valid bit pattern for `pw_time` (plain numeric fields);
    // the call only writes into it.
    let mut time: pipewire::sys::pw_time =
        unsafe { core::mem::MaybeUninit::zeroed().assume_init() };
    // SAFETY: the caller keeps the stream live for the duration of the call,
    // and the call is documented RT safe.
    let res = unsafe {
        pipewire::sys::pw_stream_get_time_n(
            stream.as_raw_ptr(),
            &mut time,
            core::mem::size_of::<pipewire::sys::pw_time>(),
        )
    };
    (res == 0).then_some(time)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(format: G2gAudioFormat, channels: u8) -> Caps {
        Caps::Audio {
            format,
            channels,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    #[test]
    fn pw_params_maps_every_opened_format_and_rejects_the_rest() {
        for (g2g, spa) in FORMATS {
            let (got, ch, rate) = pw_params(&caps(g2g, 2)).expect("pcm maps");
            assert_eq!((got, ch, rate), (spa, 2, 48_000));
        }
        assert_eq!(
            pw_params(&caps(G2gAudioFormat::Aac, 2)),
            Err(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn frame_bytes_follows_the_sample_width() {
        // S24 is 3-byte packed, so a 5.1 frame is 18 bytes.
        assert_eq!(frame_bytes(AudioFormat::S24LE, 6), 18);
        assert_eq!(frame_bytes(AudioFormat::U8, 2), 2);
        assert_eq!(frame_bytes(AudioFormat::S32LE, 2), 8);
        // a format the elements never open a stream with has no stride.
        assert_eq!(frame_bytes(AudioFormat::S16BE, 2), 0);
    }

    #[test]
    fn positions_follow_our_interleave_order() {
        let p = spa_positions(6).expect("5.1 has a layout");
        assert_eq!(
            p[..6],
            [
                spa::sys::SPA_AUDIO_CHANNEL_FL,
                spa::sys::SPA_AUDIO_CHANNEL_FR,
                spa::sys::SPA_AUDIO_CHANNEL_FC,
                spa::sys::SPA_AUDIO_CHANNEL_LFE,
                spa::sys::SPA_AUDIO_CHANNEL_RL,
                spa::sys::SPA_AUDIO_CHANNEL_RR,
            ]
        );
        // unset slots stay 0, so the tail is not read as a position.
        assert_eq!(p[6], 0);
        assert_eq!(
            spa_positions(1).unwrap()[0],
            spa::sys::SPA_AUDIO_CHANNEL_MONO
        );
        assert!(spa_positions(9).is_none());
    }
}
