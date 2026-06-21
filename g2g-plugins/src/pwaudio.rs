//! Shared PipeWire audio helpers for [`PipeWireSink`](crate::pipewiresink) and
//! [`PipeWireSrc`](crate::pipewiresrc): map our PCM `Caps` to an SPA audio
//! format pod for `Stream::connect`. Linux-only (`pipewire` feature).

use alloc::vec::Vec;

use pipewire::spa;
use spa::param::audio::{AudioFormat, AudioInfoRaw};
use spa::param::ParamType;
use spa::pod::serialize::PodSerializer;
use spa::pod::{Object, Value};
use spa::utils::SpaTypes;

use g2g_core::{AudioFormat as G2gAudioFormat, Caps, G2gError};

/// PCM parameters of an accepted `Caps::Audio`: (SPA format, channels, rate).
/// Compressed audio (AAC / Opus) is rejected structurally.
pub(crate) fn pw_params(caps: &Caps) -> Result<(AudioFormat, u32, u32), G2gError> {
    let Caps::Audio {
        format,
        channels,
        sample_rate,
    } = caps
    else {
        return Err(G2gError::CapsMismatch);
    };
    let fmt = match format {
        G2gAudioFormat::PcmS16Le => AudioFormat::S16LE,
        G2gAudioFormat::PcmF32Le => AudioFormat::F32LE,
        G2gAudioFormat::Aac | G2gAudioFormat::Opus => return Err(G2gError::CapsMismatch),
    };
    Ok((fmt, u32::from(*channels), *sample_rate))
}

/// Bytes per interleaved sample frame for an SPA audio format.
pub(crate) fn frame_bytes(format: AudioFormat, channels: u32) -> usize {
    let sample = match format {
        AudioFormat::S16LE => 2,
        AudioFormat::F32LE => 4,
        _ => 0,
    };
    sample * channels as usize
}

/// Serialize a fixed `EnumFormat` audio pod (one value) for `Stream::connect`.
/// The returned bytes back a `Pod::from_bytes` at the call site (kept there so
/// the borrow lives as long as the `connect` call needs it).
pub(crate) fn format_pod_bytes(format: AudioFormat, channels: u32, rate: u32) -> Vec<u8> {
    let mut info = AudioInfoRaw::new();
    info.set_format(format);
    info.set_rate(rate);
    info.set_channels(channels);
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
