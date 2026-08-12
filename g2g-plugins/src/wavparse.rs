//! WAV parser: a RIFF/WAVE byte stream in (`Caps::ByteStream{Wav}`), the raw
//! PCM it carries out (`Caps::Audio`), so `filesrc location=x.wav ! decodebin`
//! reaches an audio element. The read side of [`crate::wavenc::WavEnc`].
//!
//! RIFF is a chunk list: a 12-byte `RIFF....WAVE` header, then `id` + `size`
//! chunks. Only `fmt ` (the sample format) and `data` (the samples) matter here;
//! anything else (`LIST`, `fact`, a writer's own metadata) is skipped by its
//! declared size. Sizes come from the file, so they are read with checked
//! arithmetic and a chunk that overruns fails the parse.
//!
//! `data` is the last chunk this reads: its samples run to end of stream, which
//! is what a size of `0xFFFFFFFF` (the streaming sentinel `wavenc` writes) means
//! anyway.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, ANY_CHANNELS, ANY_SAMPLE_RATE,
};

/// `RIFF` + size + `WAVE`.
const RIFF_HEADER_LEN: usize = 12;
/// A chunk's 4-byte id plus its 4-byte size.
const CHUNK_HEADER_LEN: usize = 8;
/// The fields of a `fmt ` chunk this reads: tag, channels, rate, byte rate,
/// block align, bit depth.
const FMT_CHUNK_MIN_LEN: usize = 16;

const FORMAT_PCM: u16 = 1;
const FORMAT_IEEE_FLOAT: u16 = 3;
/// WAVE_FORMAT_EXTENSIBLE: the real tag sits in the chunk's extension GUID, whose
/// first two bytes are the tag it extends.
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;
/// Offset of that GUID within a `fmt ` chunk (16 fixed bytes, then `cbSize`).
const EXTENSIBLE_TAG_OFFSET: usize = 24;

/// The PCM stream a `fmt ` chunk describes.
#[derive(Debug, Clone, Copy, PartialEq)]
struct WaveFormat {
    format: AudioFormat,
    channels: u8,
    sample_rate: u32,
}

/// The g2g format for a `(wFormatTag, bits per sample)` pair, `None` for a
/// compressed or unmodeled WAV payload (ADPCM, A-law, 64-bit float).
fn audio_format(tag: u16, bits: u16) -> Option<AudioFormat> {
    match (tag, bits) {
        (FORMAT_PCM, 8) => Some(AudioFormat::PcmU8),
        (FORMAT_PCM, 16) => Some(AudioFormat::PcmS16Le),
        (FORMAT_PCM, 24) => Some(AudioFormat::PcmS24Le),
        (FORMAT_PCM, 32) => Some(AudioFormat::PcmS32Le),
        (FORMAT_IEEE_FLOAT, 32) => Some(AudioFormat::PcmF32Le),
        _ => None,
    }
}

/// Read a `fmt ` chunk body. `None` when it is too short or names a payload this
/// does not model.
fn parse_fmt(body: &[u8]) -> Option<WaveFormat> {
    if body.len() < FMT_CHUNK_MIN_LEN {
        return None;
    }
    let u16le = |o: usize| u16::from_le_bytes([body[o], body[o + 1]]);
    let u32le = |o: usize| u32::from_le_bytes([body[o], body[o + 1], body[o + 2], body[o + 3]]);
    let mut tag = u16le(0);
    if tag == FORMAT_EXTENSIBLE {
        if body.len() < EXTENSIBLE_TAG_OFFSET + 2 {
            return None;
        }
        tag = u16le(EXTENSIBLE_TAG_OFFSET);
    }
    let channels = u16le(2);
    let sample_rate = u32le(4);
    let bits = u16le(14);
    if channels == 0 || channels > u8::MAX as u16 || sample_rate == 0 {
        return None;
    }
    Some(WaveFormat {
        format: audio_format(tag, bits)?,
        channels: channels as u8,
        sample_rate,
    })
}

/// Parses a WAV byte stream into the PCM stream it carries.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::wavparse::WavParse;
///
/// let parse = WavParse::new();
/// ```
#[derive(Debug, Default)]
pub struct WavParse {
    configured: bool,
    /// Bytes accumulated across input chunks. The RIFF header and every chunk
    /// header are consumed whole, so one split across chunks stays buffered
    /// until the rest arrives.
    buf: Vec<u8>,
    riff_seen: bool,
    format: Option<WaveFormat>,
    /// True once the `data` chunk header has been consumed: everything after it
    /// is samples.
    in_data: bool,
    emitted: u64,
}

impl WavParse {
    pub fn new() -> Self {
        Self::default()
    }

    fn input_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Wav,
        }
    }

    /// The PCM formats a WAV file can carry, advertised at negotiation; the
    /// concrete one is fixed via `CapsChanged` once `fmt ` is read.
    fn output_alternatives() -> CapsSet {
        CapsSet::from_alternatives(Vec::from(
            [
                AudioFormat::PcmS16Le,
                AudioFormat::PcmF32Le,
                AudioFormat::PcmS24Le,
                AudioFormat::PcmS32Le,
                AudioFormat::PcmU8,
            ]
            .map(|format| Caps::Audio {
                format,
                channels: ANY_CHANNELS,
                sample_rate: ANY_SAMPLE_RATE,
            }),
        ))
    }

    /// Read the RIFF header and the chunks ahead of `data`, then push everything
    /// buffered after it as samples.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if !self.riff_seen {
            if self.buf.len() < RIFF_HEADER_LEN {
                return Ok(());
            }
            if &self.buf[0..4] != b"RIFF" || &self.buf[8..12] != b"WAVE" {
                return Err(G2gError::CapsMismatch);
            }
            self.buf.drain(..RIFF_HEADER_LEN);
            self.riff_seen = true;
        }
        while !self.in_data {
            if self.buf.len() < CHUNK_HEADER_LEN {
                return Ok(());
            }
            let id = [self.buf[0], self.buf[1], self.buf[2], self.buf[3]];
            let size =
                u32::from_le_bytes([self.buf[4], self.buf[5], self.buf[6], self.buf[7]]) as usize;
            if &id == b"data" {
                self.buf.drain(..CHUNK_HEADER_LEN);
                let format = self.format.ok_or(G2gError::CapsMismatch)?;
                out.push(PipelinePacket::CapsChanged(Caps::Audio {
                    format: format.format,
                    channels: format.channels,
                    sample_rate: format.sample_rate,
                }))
                .await?;
                self.in_data = true;
                break;
            }
            // A chunk body is padded to an even length, and the size comes from
            // the file: a chunk claiming more than the stream holds waits rather
            // than indexing past the buffer.
            let padded = size.checked_add(size % 2).ok_or(G2gError::CapsMismatch)?;
            let total = padded
                .checked_add(CHUNK_HEADER_LEN)
                .ok_or(G2gError::CapsMismatch)?;
            if self.buf.len() < total {
                return Ok(());
            }
            if &id == b"fmt " {
                self.format = Some(
                    parse_fmt(&self.buf[CHUNK_HEADER_LEN..CHUNK_HEADER_LEN + size])
                        .ok_or(G2gError::CapsMismatch)?,
                );
            }
            self.buf.drain(..total);
        }
        if self.buf.is_empty() {
            return Ok(());
        }
        let samples = core::mem::take(&mut self.buf);
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(samples.into_boxed_slice())),
            FrameTiming::default(),
            self.emitted,
        );
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}

impl AsyncElement for WavParse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "WAV parser",
            "Codec/Demuxer/Audio",
            "Reads the PCM stream out of a RIFF/WAVE byte stream",
            "g2g",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    /// The layout is unknown until the `fmt ` chunk is read, so this advertises
    /// what the audio decoder does (M754): a concrete default rate first, the
    /// `ANY_SAMPLE_RATE` wildcard second so a downstream `rate=` pin intersects
    /// to it. A lone wildcard cannot fixate. The real rate, channel count and
    /// format arrive with the `CapsChanged` the parsed header emits.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Wav,
            } => {
                let pcm = |sample_rate| Caps::Audio {
                    format: AudioFormat::PcmS16Le,
                    channels: ANY_CHANNELS,
                    sample_rate,
                };
                CapsSet::from_alternatives(alloc::vec![pcm(48_000), pcm(ANY_SAMPLE_RATE)])
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Wav
            }
        ) {
            return Err(G2gError::CapsMismatch);
        }
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
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    self.buf.extend_from_slice(slice);
                    self.drain(out).await?;
                }
                PipelinePacket::Eos => {
                    self.drain(out).await?;
                }
                // The byte stream carries no sample format; the concrete caps
                // come from the parsed `fmt ` chunk instead.
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for WavParse {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(Self::output_alternatives()),
        ])
    }
}
