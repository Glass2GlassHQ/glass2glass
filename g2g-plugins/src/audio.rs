//! Shared PCM / `WAVEFORMATEX` helpers for the audio sinks and sources
//! (`WavSink`, `WasapiSink`, `WasapiSrc`). std-gated like its callers.

use g2g_core::{AudioFormat, Caps, G2gError};

/// `WAVEFORMATEX` format tags.
pub(crate) const WAVE_FORMAT_PCM: u16 = 1;
pub(crate) const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;

/// PCM parameters of an accepted caps: (format tag, bits, channels, rate).
/// Compressed audio (AAC/Opus) is rejected structurally.
pub(crate) fn pcm_params(caps: &Caps) -> Result<(u16, u16, u16, u32), G2gError> {
    let Caps::Audio {
        format,
        channels,
        sample_rate,
    } = caps
    else {
        return Err(G2gError::CapsMismatch);
    };
    let (tag, bits) = match format {
        AudioFormat::PcmS16Le => (WAVE_FORMAT_PCM, 16u16),
        AudioFormat::PcmF32Le => (WAVE_FORMAT_IEEE_FLOAT, 32u16),
        AudioFormat::Aac | AudioFormat::Opus => return Err(G2gError::CapsMismatch),
    };
    Ok((tag, bits, *channels as u16, *sample_rate))
}
