//! WAV file sink (M25). Writes interleaved PCM (`PcmS16Le` or `PcmF32Le`)
//! to a standard RIFF/WAVE file, so an audio pipeline's output is playable
//! anywhere. The header's running sizes are patched in place on `Eos`
//! (WAV is not stream-friendly; the fragmented recording format for live
//! durability remains `Mp4Sink` on the video side).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::PathBuf;

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

use crate::filesink::io_err;

/// Byte offsets of the two running sizes in the 44-byte canonical header.
const RIFF_SIZE_OFFSET: u64 = 4;
const DATA_SIZE_OFFSET: u64 = 40;

#[derive(Debug)]
pub struct WavSink {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    data_bytes: u64,
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

/// The canonical 44-byte header with zeroed running sizes.
fn wav_header(tag: u16, bits: u16, channels: u16, rate: u32) -> Vec<u8> {
    let block_align = channels * bits / 8;
    let byte_rate = rate * block_align as u32;
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&0u32.to_le_bytes()); // riff size, patched at Eos
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    h.extend_from_slice(&tag.to_le_bytes());
    h.extend_from_slice(&channels.to_le_bytes());
    h.extend_from_slice(&rate.to_le_bytes());
    h.extend_from_slice(&byte_rate.to_le_bytes());
    h.extend_from_slice(&block_align.to_le_bytes());
    h.extend_from_slice(&bits.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&0u32.to_le_bytes()); // data size, patched at Eos
    h
}

impl AsyncElement for WavSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
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
        let file = File::create(&self.path).map_err(io_err)?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(&wav_header(tag, bits, channels, rate))
            .map_err(io_err)?;
        self.writer = Some(writer);
        self.data_bytes = 0;
        Ok(ConfigureOutcome::Accepted)
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
                    let riff_size = (36 + self.data_bytes) as u32;
                    file.seek(SeekFrom::Start(RIFF_SIZE_OFFSET)).map_err(io_err)?;
                    file.write_all(&riff_size.to_le_bytes()).map_err(io_err)?;
                    file.seek(SeekFrom::Start(DATA_SIZE_OFFSET)).map_err(io_err)?;
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
            }
            Ok(())
        })
    }
}

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
