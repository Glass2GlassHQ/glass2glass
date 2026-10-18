use alloc::vec::Vec;

use g2g_core::AudioFormat;

use crate::riff::{chunk_len, put_chunk, put_chunk_header, FourCc, RIFF_FOURCC, RIFF_HEADER_LEN};

pub(crate) const WAVE_FORMAT_PCM: u16 = 0x0001;
pub(crate) const WAVE_FORMAT_IEEE_FLOAT: u16 = 0x0003;
const WAVE_FORMAT_ALAW: u16 = 0x0006;
const WAVE_FORMAT_MULAW: u16 = 0x0007;
const WAVE_FORMAT_IMA_ADPCM: u16 = 0x0011;
const WAVE_FORMAT_MP3: u16 = 0x0055;
const WAVE_FORMAT_AAC: u16 = 0x00FF;
const WAVE_FORMAT_AC3: u16 = 0x2000;

/// The `wBitsPerSample` of a compressed payload, which has no sample depth. A
/// reader takes any value for it.
const COMPRESSED_BITS: u16 = 0;
const G711_BITS: u16 = 8;
const IMA_ADPCM_BITS: u16 = 4;

/// Each format a `WAVEFORMATEX` can name: its `wFormatTag` and
/// `wBitsPerSample`.
const WAVE_FORMATS: [(AudioFormat, u16, u16); 11] = [
    (AudioFormat::PcmU8, WAVE_FORMAT_PCM, 8),
    (AudioFormat::PcmS16Le, WAVE_FORMAT_PCM, 16),
    (AudioFormat::PcmS24Le, WAVE_FORMAT_PCM, 24),
    (AudioFormat::PcmS32Le, WAVE_FORMAT_PCM, 32),
    (AudioFormat::PcmF32Le, WAVE_FORMAT_IEEE_FLOAT, 32),
    (AudioFormat::Alaw, WAVE_FORMAT_ALAW, G711_BITS),
    (AudioFormat::Mulaw, WAVE_FORMAT_MULAW, G711_BITS),
    (AudioFormat::ImaAdpcm, WAVE_FORMAT_IMA_ADPCM, IMA_ADPCM_BITS),
    (AudioFormat::Mp3, WAVE_FORMAT_MP3, COMPRESSED_BITS),
    (AudioFormat::Ac3, WAVE_FORMAT_AC3, COMPRESSED_BITS),
    (AudioFormat::Aac, WAVE_FORMAT_AAC, COMPRESSED_BITS),
];

/// `(wFormatTag, wBitsPerSample)` for a format, `None` for one no
/// `WAVEFORMATEX` names.
pub(crate) fn wave_format_tag(format: AudioFormat) -> Option<(u16, u16)> {
    WAVE_FORMATS
        .iter()
        .find(|(known, _, _)| *known == format)
        .map(|(_, tag, bits)| (*tag, *bits))
}

/// The format a `(wFormatTag, wBitsPerSample)` pair names, `None` for a payload
/// g2g does not model.
pub(crate) fn audio_format(tag: u16, bits: u16) -> Option<AudioFormat> {
    WAVE_FORMATS
        .iter()
        .find(|(_, known_tag, known_bits)| {
            *known_tag == tag && (*known_bits == bits || *known_bits == COMPRESSED_BITS)
        })
        .map(|(format, _, _)| *format)
}

// how a decode error names a `wFormatTag` g2g does not decode
pub(crate) fn format_tag_name(tag: u16) -> alloc::string::String {
    alloc::format!("audio format tag {tag:#06x}")
}

/// Bytes one sample frame of a constant-width format takes, `None` for a
/// compressed or block-coded one.
pub(crate) fn sample_frame_len(format: AudioFormat, channels: u8) -> Option<usize> {
    let (tag, bits) = wave_format_tag(format)?;
    let constant_width = matches!(
        tag,
        WAVE_FORMAT_PCM | WAVE_FORMAT_IEEE_FLOAT | WAVE_FORMAT_ALAW | WAVE_FORMAT_MULAW
    );
    constant_width.then(|| usize::from(bits / 8) * usize::from(channels.max(1)))
}

/// The fields of a `WAVEFORMATEX`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WaveFormatEx {
    pub(crate) tag: u16,
    pub(crate) channels: u16,
    pub(crate) sample_rate: u32,
    pub(crate) byte_rate: u32,
    pub(crate) block_align: u16,
    pub(crate) bits: u16,
}

impl WaveFormatEx {
    /// The 16-byte `PCMWAVEFORMAT` when `extension` is `None`, else the full
    /// `WAVEFORMATEX`: `cbSize` then the extension.
    pub(crate) fn to_bytes(self, extension: Option<&[u8]>) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.tag.to_le_bytes());
        out.extend_from_slice(&self.channels.to_le_bytes());
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&self.byte_rate.to_le_bytes());
        out.extend_from_slice(&self.block_align.to_le_bytes());
        out.extend_from_slice(&self.bits.to_le_bytes());
        if let Some(extension) = extension {
            out.extend_from_slice(&(extension.len() as u16).to_le_bytes());
            out.extend_from_slice(extension);
        }
        out
    }
}

const WAVE_FOURCC: FourCc = *b"WAVE";
const FMT_FOURCC: FourCc = *b"fmt ";
const FACT_FOURCC: FourCc = *b"fact";
const DATA_FOURCC: FourCc = *b"data";

/// A WAVE file's header, ending with the `data` chunk header. Plain PCM gets the
/// canonical 44 bytes. Any other tag gets a `WAVEFORMATEX` carrying `extension`
/// and the `fact` chunk the spec requires of it, holding `sample_count`.
pub(crate) fn wav_header(
    format: &WaveFormatEx,
    extension: &[u8],
    riff_len: u32,
    sample_count: u32,
    data_len: u32,
) -> Vec<u8> {
    let pcm = format.tag == WAVE_FORMAT_PCM;
    let fmt = format.to_bytes((!pcm).then_some(extension));
    let mut out = Vec::with_capacity(
        RIFF_HEADER_LEN + chunk_len(fmt.len()) + chunk_len(FACT_LEN) + chunk_len(0),
    );
    out.extend_from_slice(&RIFF_FOURCC);
    out.extend_from_slice(&riff_len.to_le_bytes());
    out.extend_from_slice(&WAVE_FOURCC);
    put_chunk(&mut out, &FMT_FOURCC, &fmt);
    if !pcm {
        put_chunk(&mut out, &FACT_FOURCC, &sample_count.to_le_bytes());
    }
    put_chunk_header(&mut out, &DATA_FOURCC, data_len as usize);
    out
}

/// A `fact` body: one sample count.
const FACT_LEN: usize = 4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tag_names_its_format_back() {
        for (format, tag, bits) in WAVE_FORMATS {
            assert_eq!(wave_format_tag(format), Some((tag, bits)));
            assert_eq!(audio_format(tag, bits), Some(format), "{format:?}");
        }
    }

    #[test]
    fn a_non_pcm_header_carries_cb_size_and_fact() {
        const SAMPLE_COUNT: u32 = 1234;
        let alaw = WaveFormatEx {
            tag: WAVE_FORMAT_ALAW,
            channels: 1,
            sample_rate: 8_000,
            byte_rate: 8_000,
            block_align: 1,
            bits: G711_BITS,
        };
        let header = wav_header(&alaw, &[], 0, SAMPLE_COUNT, 0);
        let fmt_len = u32::from_le_bytes(header[16..20].try_into().unwrap());
        assert_eq!(fmt_len, 18, "WAVEFORMATEX with cbSize");
        let fact = 20 + fmt_len as usize;
        assert_eq!(&header[fact..fact + 4], b"fact");
        assert_eq!(
            u32::from_le_bytes(header[fact + 8..fact + 12].try_into().unwrap()),
            SAMPLE_COUNT
        );
        assert_eq!(&header[header.len() - 8..header.len() - 4], b"data");
    }
}
