//! WASAPI pieces shared by the render sink, the capture source, and the
//! endpoint provider (the Windows sibling of [`alsapcm`](crate::alsapcm)):
//! selecting an endpoint by id, and reading a `WAVEFORMATEX` as the PCM shape
//! g2g carries.
//!
//! Selection lives here so the id [`wasapidevice`](crate::wasapidevice) reports
//! and the id the elements' `device=` accepts cannot drift apart.

use alloc::vec::Vec;

use windows::core::PCWSTR;
use windows::Win32::Media::Audio::{
    eConsole, EDataFlow, IMMDevice, IMMDeviceEnumerator, MMDeviceEnumerator, WAVEFORMATEX,
};
use windows::Win32::System::Com::{CoCreateInstance, CLSCTX_INPROC_SERVER};

use g2g_core::{AudioFormat, G2gError, HardwareError};

use crate::audio::{WAVE_FORMAT_IEEE_FLOAT, WAVE_FORMAT_PCM};

/// `WAVE_FORMAT_EXTENSIBLE`: the tag a mix format usually carries, wrapping
/// the real subtype in a `WAVEFORMATEXTENSIBLE` tail.
pub(crate) const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// An endpoint's PCM shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AudioConfig {
    pub(crate) format: AudioFormat,
    pub(crate) channels: u8,
    pub(crate) sample_rate: u32,
    pub(crate) block_align: usize,
}

/// WASAPI errors are COM HRESULTs, the same carrier as the MF path.
pub(crate) fn audio_err(e: windows::core::Error) -> G2gError {
    G2gError::Hardware(HardwareError::MediaFoundation(e.code().0))
}

/// The endpoint `id` names, or the default console endpoint for `dataflow`
/// when `id` is empty.
///
/// # Safety
/// Must run on a COM-initialised thread.
pub(crate) unsafe fn open_endpoint(dataflow: EDataFlow, id: &str) -> Result<IMMDevice, G2gError> {
    // SAFETY: COM object creation / queries on the owning thread. The wide id
    // buffer outlives the GetDevice call that borrows it.
    unsafe {
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_INPROC_SERVER).map_err(audio_err)?;
        if id.is_empty() {
            return enumerator
                .GetDefaultAudioEndpoint(dataflow, eConsole)
                .map_err(audio_err);
        }
        let wide = wide_z(id);
        enumerator
            .GetDevice(PCWSTR(wide.as_ptr()))
            .map_err(audio_err)
    }
}

/// A NUL-terminated UTF-16 copy of `text`, for the Win32 string parameters.
pub(crate) fn wide_z(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(core::iter::once(0)).collect()
}

/// Map a `WAVEFORMATEX` to an [`AudioConfig`]. 32-bit samples are reported as
/// float (the shared-mode mix format), 16-bit as signed PCM; other depths are
/// unsupported.
pub(crate) fn audio_config_from_format(fmt: &WAVEFORMATEX) -> Result<AudioConfig, G2gError> {
    let format = match (fmt.wFormatTag, fmt.wBitsPerSample) {
        (WAVE_FORMAT_PCM, 16) => AudioFormat::PcmS16Le,
        (WAVE_FORMAT_IEEE_FLOAT, 32) => AudioFormat::PcmF32Le,
        // EXTENSIBLE wraps the real subtype; the mix format is in practice
        // 32-bit float or 16-bit PCM, so map by bit depth.
        (WAVE_FORMAT_EXTENSIBLE, 32) => AudioFormat::PcmF32Le,
        (WAVE_FORMAT_EXTENSIBLE, 16) => AudioFormat::PcmS16Le,
        _ => return Err(G2gError::CapsMismatch),
    };
    Ok(AudioConfig {
        format,
        channels: u8::try_from(fmt.nChannels).map_err(|_| G2gError::CapsMismatch)?,
        sample_rate: fmt.nSamplesPerSec,
        block_align: fmt.nBlockAlign as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format(tag: u16, bits: u16, channels: u16, rate: u32) -> WAVEFORMATEX {
        WAVEFORMATEX {
            wFormatTag: tag,
            nChannels: channels,
            nSamplesPerSec: rate,
            nAvgBytesPerSec: rate * u32::from(channels) * u32::from(bits / 8),
            nBlockAlign: channels * (bits / 8),
            wBitsPerSample: bits,
            cbSize: 0,
        }
    }

    #[test]
    fn mix_formats_map_to_the_pcm_shapes_the_elements_carry() {
        assert_eq!(
            audio_config_from_format(&format(WAVE_FORMAT_EXTENSIBLE, 32, 2, 48_000)),
            Ok(AudioConfig {
                format: AudioFormat::PcmF32Le,
                channels: 2,
                sample_rate: 48_000,
                block_align: 8,
            })
        );
        assert_eq!(
            audio_config_from_format(&format(WAVE_FORMAT_PCM, 16, 1, 44_100))
                .expect("pcm")
                .format,
            AudioFormat::PcmS16Le
        );
        // a depth neither element carries is rejected, not rounded.
        assert_eq!(
            audio_config_from_format(&format(WAVE_FORMAT_PCM, 24, 2, 48_000)),
            Err(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn an_id_becomes_a_nul_terminated_wide_string() {
        assert_eq!(wide_z("hi"), [0x68, 0x69, 0]);
        assert_eq!(wide_z(""), [0]);
    }
}
