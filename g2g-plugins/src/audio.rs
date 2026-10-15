//! Shared PCM helpers for the audio sinks and sources: the `WAVEFORMATEX`
//! mapping (`WavSink`, `WasapiSink`, `WasapiSrc`) and the PCM accept set the
//! audio sinks declare. std-gated like its callers.

use g2g_core::{pcm_formats, AudioFormat, Caps, CapsSet, G2gError, ANY_CHANNELS, ANY_SAMPLE_RATE};

use crate::waveformat::wave_format_tag;

pub(crate) use crate::waveformat::{WAVE_FORMAT_IEEE_FLOAT, WAVE_FORMAT_PCM};

/// PCM parameters of an accepted caps: (format tag, bits, channels, rate).
/// Compressed audio (AAC/Opus) is rejected structurally.
pub(crate) fn pcm_params(caps: &Caps) -> Result<(u16, u16, u16, u32), G2gError> {
    let Caps::Audio {
        format,
        channels,
        sample_rate,
        ..
    } = caps
    else {
        return Err(G2gError::CapsMismatch);
    };
    if *channels == ANY_CHANNELS || *sample_rate == ANY_SAMPLE_RATE {
        return Err(G2gError::CapsMismatch);
    }
    let (tag, bits) = linear_pcm_tag(*format).ok_or(G2gError::CapsMismatch)?;
    Ok((tag, bits, *channels as u16, *sample_rate))
}

// Only linear PCM, integer or float, is what these devices and files take.
fn linear_pcm_tag(format: AudioFormat) -> Option<(u16, u16)> {
    wave_format_tag(format)
        .filter(|(tag, _)| matches!(*tag, WAVE_FORMAT_PCM | WAVE_FORMAT_IEEE_FLOAT))
}

pub(crate) fn linear_pcm_caps() -> CapsSet {
    pcm_caps_any_rate_and_channels(|format| linear_pcm_tag(format).is_some())
}

pub(crate) fn pcm_caps_any_rate_and_channels(
    takes_format: impl Fn(AudioFormat) -> bool,
) -> CapsSet {
    let alternatives = pcm_formats()
        .into_iter()
        .filter(|format| takes_format(*format))
        .map(|format| Caps::Audio {
            format,
            channels: ANY_CHANNELS,
            sample_rate: ANY_SAMPLE_RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .collect();
    CapsSet::from_alternatives(alternatives)
}
