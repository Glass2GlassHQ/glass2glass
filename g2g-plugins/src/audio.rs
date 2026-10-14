//! Shared PCM / `WAVEFORMATEX` helpers for the audio sinks and sources
//! (`WavSink`, `WasapiSink`, `WasapiSrc`). std-gated like its callers.

use g2g_core::{Caps, G2gError};

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
    // Only linear PCM, integer or float, is what these devices and files take.
    let (tag, bits) = wave_format_tag(*format)
        .filter(|(tag, _)| matches!(*tag, WAVE_FORMAT_PCM | WAVE_FORMAT_IEEE_FLOAT))
        .ok_or(G2gError::CapsMismatch)?;
    Ok((tag, bits, *channels as u16, *sample_rate))
}
