//! WAV file sink (M25). Writes interleaved PCM (`PcmS16Le` or `PcmF32Le`)
//! to a standard RIFF/WAVE file, so an audio pipeline's output is playable
//! anywhere. The header's running sizes are patched in place on `Eos`
//! (WAV is not stream-friendly; the fragmented recording format for live
//! durability remains `Mp4Mux` on the video side).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::PathBuf;

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

use crate::audio::WAVE_FORMAT_IEEE_FLOAT;
use crate::filesink::io_err;

/// Byte offset of the RIFF running size (constant across header layouts).
const RIFF_SIZE_OFFSET: u64 = 4;

#[derive(Debug)]
pub struct WavSink {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    data_bytes: u64,
    /// Length of the written header, so the running sizes (always its last
    /// chunk) are patched at the right offsets for either layout.
    header_len: u64,
    /// Frame size in bytes, for the non-PCM `fact` chunk's sample count.
    block_align: u32,
    eos_seen: bool,
}

impl WavSink {
    /// The file is created in `configure_pipeline`; construction has no
    /// filesystem side effects.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            writer: None,
            data_bytes: 0,
            header_len: 0,
            block_align: 0,
            eos_seen: false,
        }
    }

    pub fn bytes_written(&self) -> u64 {
        self.data_bytes
    }

    pub fn eos_seen(&self) -> bool {
        self.eos_seen
    }
}

use crate::audio::pcm_params;

/// The header with zeroed running sizes. PCM gets the canonical 44-byte form;
/// IEEE float gets the spec-required 18-byte fmt chunk (with `cbSize`) plus a
/// `fact` chunk, so non-PCM output is conformant, not just widely accepted.
fn wav_header(tag: u16, bits: u16, channels: u16, rate: u32) -> Vec<u8> {
    let block_align = channels * bits / 8;
    let byte_rate = rate * block_align as u32;
    let is_float = tag == WAVE_FORMAT_IEEE_FLOAT;
    let mut h = Vec::with_capacity(if is_float { 58 } else { 44 });
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&0u32.to_le_bytes()); // riff size, patched at Eos
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&if is_float { 18u32 } else { 16 }.to_le_bytes());
    h.extend_from_slice(&tag.to_le_bytes());
    h.extend_from_slice(&channels.to_le_bytes());
    h.extend_from_slice(&rate.to_le_bytes());
    h.extend_from_slice(&byte_rate.to_le_bytes());
    h.extend_from_slice(&block_align.to_le_bytes());
    h.extend_from_slice(&bits.to_le_bytes());
    if is_float {
        h.extend_from_slice(&0u16.to_le_bytes()); // cbSize = 0
        h.extend_from_slice(b"fact");
        h.extend_from_slice(&4u32.to_le_bytes());
        h.extend_from_slice(&0u32.to_le_bytes()); // sample count, patched at Eos
    }
    h.extend_from_slice(b"data");
    h.extend_from_slice(&0u32.to_le_bytes()); // data size, patched at Eos
    h
}

impl AsyncElement for WavSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        pcm_params(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// PCM only; `Caps::Audio` has no open dims, so the legacy intercept
    /// bridge carries the per-rate/channel acceptance.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::LegacySink(Box::new(|c: &Caps| {
            pcm_params(c)?;
            Ok(c.clone())
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (tag, bits, channels, rate) = pcm_params(absolute_caps)?;
        let header = wav_header(tag, bits, channels, rate);
        self.header_len = header.len() as u64;
        self.block_align = (channels * bits / 8) as u32;
        let file = File::create(&self.path).map_err(io_err)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&header).map_err(io_err)?;
        self.writer = Some(writer);
        self.data_bytes = 0;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        WAVSINK_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "WAV file sink",
            "Sink/File",
            "Writes interleaved PCM to a RIFF/WAVE file",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.path = PathBuf::from(value.as_str().ok_or(PropError::Type)?);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.path.to_string_lossy().into_owned())),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let writer = self.writer.as_mut().ok_or(G2gError::NotConfigured)?;
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    writer.write_all(slice.as_slice()).map_err(io_err)?;
                    self.data_bytes += slice.as_slice().len() as u64;
                }
                PipelinePacket::Eos => {
                    // patch the running sizes, then return to the end so a
                    // (non-standard) post-Eos write stays appendable.
                    writer.flush().map_err(io_err)?;
                    let file = writer.get_mut();
                    // The data chunk's size word is always the last 4 bytes of
                    // the header; riff size is everything after the RIFF+size.
                    let riff_size = (self.header_len - 8 + self.data_bytes) as u32;
                    file.seek(SeekFrom::Start(RIFF_SIZE_OFFSET))
                        .map_err(io_err)?;
                    file.write_all(&riff_size.to_le_bytes()).map_err(io_err)?;
                    // The non-PCM `fact` chunk (when present) carries the per-
                    // channel sample count, 12 bytes before the data size word.
                    if self.header_len > 44 && self.block_align > 0 {
                        let samples = (self.data_bytes / self.block_align as u64) as u32;
                        file.seek(SeekFrom::Start(self.header_len - 12))
                            .map_err(io_err)?;
                        file.write_all(&samples.to_le_bytes()).map_err(io_err)?;
                    }
                    file.seek(SeekFrom::Start(self.header_len - 4))
                        .map_err(io_err)?;
                    file.write_all(&(self.data_bytes as u32).to_le_bytes())
                        .map_err(io_err)?;
                    file.seek(SeekFrom::End(0)).map_err(io_err)?;
                    file.flush().map_err(io_err)?;
                    self.eos_seen = true;
                }
                // a mid-stream format change can't be expressed in a WAV
                // header; only a caps identical to the configured one passes.
                PipelinePacket::CapsChanged(c) => {
                    pcm_params(&c)?;
                }
                PipelinePacket::Flush => {}
                // Segment is control: ignored at sink.
                PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
            }
            Ok(())
        })
    }
}

/// `WavSink`'s settable properties: the output file path.
static WAVSINK_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "location",
    PropKind::Str,
    "output file path",
)];

impl PadTemplates for WavSink {
    /// Terminal PCM sink pad. `Caps::Audio` has no open dims, so the
    /// template pins the common shapes per PCM format.
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = |format| Caps::Audio {
            format,
            channels: 2,
            sample_rate: 48_000,
        };
        Vec::from([PadTemplate::sink(CapsSet::from_alternatives(Vec::from([
            pcm(AudioFormat::PcmS16Le),
            pcm(AudioFormat::PcmF32Le),
        ])))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_canonical_44_bytes() {
        let h = wav_header(1, 16, 2, 48_000);
        assert_eq!(h.len(), 44);
        assert_eq!(&h[..4], b"RIFF");
        assert_eq!(&h[8..12], b"WAVE");
        assert_eq!(&h[12..16], b"fmt ");
        assert_eq!(&h[36..40], b"data");
        // block align 4 (stereo s16), byte rate 192000
        assert_eq!(u16::from_le_bytes(h[32..34].try_into().unwrap()), 4);
        assert_eq!(u32::from_le_bytes(h[28..32].try_into().unwrap()), 192_000);
    }

    #[test]
    fn float_header_has_18_byte_fmt_and_fact_chunk() {
        let h = wav_header(WAVE_FORMAT_IEEE_FLOAT, 32, 2, 48_000);
        assert_eq!(h.len(), 58);
        assert_eq!(&h[12..16], b"fmt ");
        assert_eq!(
            u32::from_le_bytes(h[16..20].try_into().unwrap()),
            18,
            "18-byte fmt for float"
        );
        assert_eq!(
            u16::from_le_bytes(h[36..38].try_into().unwrap()),
            0,
            "cbSize = 0"
        );
        assert_eq!(&h[38..42], b"fact");
        assert_eq!(&h[50..54], b"data");
    }

    #[tokio::test]
    async fn float_wav_patches_fact_and_data_sizes() {
        use g2g_core::{Frame, FrameTiming, PushOutcome, SystemSlice};
        struct NullSink;
        impl OutputSink for NullSink {
            fn push<'a>(
                &'a mut self,
                _p: PipelinePacket,
            ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
                Box::pin(async { Ok(PushOutcome::Accepted) })
            }
        }
        let path = std::env::temp_dir().join("g2g_wavsink_float.wav");
        let _ = std::fs::remove_file(&path);
        let mut sink = WavSink::new(&path);
        let caps = Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 2,
            sample_rate: 48_000,
        };
        sink.configure_pipeline(&caps).unwrap();
        let mut out = NullSink;
        // 8 f32 = 32 bytes = 4 stereo frames (block_align 8).
        let data: Vec<u8> = [0.5f32; 8]
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        sink.process(PipelinePacket::DataFrame(frame), &mut out)
            .await
            .unwrap();
        sink.process(PipelinePacket::Eos, &mut out).await.unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 58 + 32);
        assert_eq!(&bytes[38..42], b"fact");
        assert_eq!(
            u32::from_le_bytes(bytes[46..50].try_into().unwrap()),
            4,
            "4 samples/channel"
        );
        assert_eq!(
            u32::from_le_bytes(bytes[54..58].try_into().unwrap()),
            32,
            "data size"
        );
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            82,
            "riff size"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn compressed_audio_is_rejected() {
        let aac = Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
        };
        assert_eq!(pcm_params(&aac), Err(G2gError::CapsMismatch));
        let f32le = Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 1,
            sample_rate: 44_100,
        };
        assert_eq!(pcm_params(&f32le), Ok((3, 32, 1, 44_100)));
    }
}
