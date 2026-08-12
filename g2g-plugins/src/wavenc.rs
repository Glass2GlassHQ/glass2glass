//! WAV encoder: raw PCM in (`Caps::Audio`), a RIFF/WAVE byte stream out
//! (`Caps::ByteStream{Wav}`), so a separated or synthesized track lands as a
//! file an audio tool opens.
//!
//! ```text
//! ... ! audioconvert ! wavenc ! filesink location=out.wav
//! ```
//!
//! The 44-byte canonical header goes out ahead of the first sample, with the two
//! size fields written as the streaming sentinel `0xFFFFFFFF` (read the data to
//! end of file). A g2g sink writes forward only, so there is no seek back to
//! patch the real sizes in at end of stream, and a sentinel is what a reader
//! copes with; a zero would have it believe the file is empty. Samples then pass
//! through untouched: WAV is a header plus the PCM as it arrived.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

/// Size fields of a stream whose length is not known when the header is written.
/// A reader takes the `data` chunk as running to the end of the file.
const STREAMING_SIZE: u32 = u32::MAX;

/// `wFormatTag` values: integer PCM, and IEEE float for the 32-bit float input.
const FORMAT_PCM: u16 = 1;
const FORMAT_IEEE_FLOAT: u16 = 3;

/// The PCM inputs a WAV file can carry.
fn input_alternatives() -> Vec<Caps> {
    Vec::from(
        [
            AudioFormat::PcmS16Le,
            AudioFormat::PcmF32Le,
            AudioFormat::PcmS24Le,
            AudioFormat::PcmS32Le,
            AudioFormat::PcmU8,
        ]
        .map(|format| Caps::Audio {
            format,
            channels: 0,
            sample_rate: 0,
        }),
    )
}

/// `(wFormatTag, bits per sample)` of a PCM format, `None` for anything WAV
/// cannot describe in a `fmt ` chunk this way.
fn wave_format(format: AudioFormat) -> Option<(u16, u16)> {
    match format {
        AudioFormat::PcmU8 => Some((FORMAT_PCM, 8)),
        AudioFormat::PcmS16Le => Some((FORMAT_PCM, 16)),
        AudioFormat::PcmS24Le => Some((FORMAT_PCM, 24)),
        AudioFormat::PcmS32Le => Some((FORMAT_PCM, 32)),
        AudioFormat::PcmF32Le => Some((FORMAT_IEEE_FLOAT, 32)),
        _ => None,
    }
}

/// The 44-byte canonical RIFF/WAVE header for this stream.
fn header(format: AudioFormat, channels: u16, sample_rate: u32) -> Option<Vec<u8>> {
    let (tag, bits) = wave_format(format)?;
    let block_align = channels * bits / 8;
    let byte_rate = sample_rate * block_align as u32;
    let mut out = Vec::with_capacity(44);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&STREAMING_SIZE.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM `fmt ` chunk size
    out.extend_from_slice(&tag.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&STREAMING_SIZE.to_le_bytes());
    Some(out)
}

/// The WAV byte stream this element produces.
fn wav_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::Wav,
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::wavenc::WavEnc;
///
/// let element = WavEnc::new();
/// ```
#[derive(Debug, Default)]
pub struct WavEnc {
    /// The negotiated PCM stream, captured at configure time so the header can
    /// describe it.
    input: Option<(AudioFormat, u16, u32)>,
    header_written: bool,
    configured: bool,
    emitted: u64,
}

impl WavEnc {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count of packets pushed downstream, the header included.
    pub fn emitted_count(&self) -> u64 {
        self.emitted
    }

    fn frame(&mut self, bytes: Vec<u8>) -> PipelinePacket {
        self.emitted += 1;
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: Default::default(),
            sequence: self.emitted,
            meta: Default::default(),
        })
    }
}

impl AsyncElement for WavEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn is_format_boundary(&self) -> bool {
        true
    }

    /// The input side: the PCM this can describe in a `fmt ` chunk, at the rate
    /// and channel count upstream negotiated.
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio { format, .. } if wave_format(*format).is_some() => {
                Ok(upstream_caps.clone())
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio { format, .. } if wave_format(*format).is_some() => {
                CapsSet::one(wav_caps())
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::Audio {
            format,
            channels,
            sample_rate,
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if wave_format(*format).is_none() || *channels == 0 || *sample_rate == 0 {
            return Err(G2gError::CapsMismatch);
        }
        self.input = Some((*format, *channels as u16, *sample_rate));
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let (format, channels, sample_rate) =
                        self.input.ok_or(G2gError::NotConfigured)?;
                    if !self.header_written {
                        out.push(PipelinePacket::CapsChanged(wav_caps())).await?;
                        let bytes =
                            header(format, channels, sample_rate).ok_or(G2gError::CapsMismatch)?;
                        let packet = self.frame(bytes);
                        out.push(packet).await?;
                        self.header_written = true;
                    }
                    let samples = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let packet = self.frame(samples.to_vec());
                    out.push(packet).await?;
                }
                // The input caps describe PCM; this element's output is the WAV
                // stream it already announced, so an input caps change is not
                // forwarded. A different PCM format mid-stream would need a new
                // header, which a written file cannot take.
                PipelinePacket::CapsChanged(_) => {}
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "WAV encoder",
            "Codec/Muxer/Audio",
            "Wraps raw PCM in a RIFF/WAVE byte stream",
            "g2g",
        )
    }
}

impl PadTemplates for WavEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(input_alternatives())),
            PadTemplate::source(CapsSet::one(wav_caps())),
        ])
    }
}
