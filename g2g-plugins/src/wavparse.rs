//! WAV parser: a RIFF/WAVE byte stream in (`Caps::ByteStream{Wav}`), the audio
//! it carries out (`Caps::Audio`), so `filesrc location=x.wav ! decodebin`
//! reaches an audio element. The read side of [`crate::wavenc::WavEnc`]. A
//! G.711 or IMA ADPCM payload comes out coded, for [`crate::g711`] /
//! [`crate::adpcm`] to decode.
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
    pcm_formats, AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet,
    ConfigureOutcome, ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, ANY_CHANNELS, ANY_SAMPLE_RATE,
};

use crate::riff::{
    padded_len, read_fourcc, read_u16, read_u32, CHUNK_HEADER_LEN, FOURCC_LEN, RIFF_FOURCC,
    RIFF_HEADER_LEN,
};

/// The fields of a `fmt ` chunk this reads: tag, channels, rate, byte rate,
/// block align, bit depth.
const FMT_CHUNK_MIN_LEN: usize = 16;

const FORMAT_PCM: u16 = 1;
const FORMAT_IEEE_FLOAT: u16 = 3;
const FORMAT_ALAW: u16 = 6;
const FORMAT_MULAW: u16 = 7;
const FORMAT_IMA_ADPCM: u16 = 0x11;
/// Bits per sample of the companded G.711 payloads.
const G711_BITS: u16 = 8;
/// Bits per sample of an IMA ADPCM nibble.
const IMA_ADPCM_BITS: u16 = 4;
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

/// The g2g format for a `(wFormatTag, bits per sample)` pair, `None` for an
/// unmodeled WAV payload (64-bit float, MS ADPCM).
fn audio_format(tag: u16, bits: u16) -> Option<AudioFormat> {
    match (tag, bits) {
        (FORMAT_PCM, 8) => Some(AudioFormat::PcmU8),
        (FORMAT_PCM, 16) => Some(AudioFormat::PcmS16Le),
        (FORMAT_PCM, 24) => Some(AudioFormat::PcmS24Le),
        (FORMAT_PCM, 32) => Some(AudioFormat::PcmS32Le),
        (FORMAT_IEEE_FLOAT, 32) => Some(AudioFormat::PcmF32Le),
        (FORMAT_ALAW, G711_BITS) => Some(AudioFormat::Alaw),
        (FORMAT_MULAW, G711_BITS) => Some(AudioFormat::Mulaw),
        (FORMAT_IMA_ADPCM, IMA_ADPCM_BITS) => Some(AudioFormat::ImaAdpcm),
        _ => None,
    }
}

/// The coded payloads a WAV file can carry, at the unknown-layout sentinels a
/// decoder negotiates against before the `fmt ` chunk is read.
fn coded_alternatives() -> [Caps; 3] {
    [AudioFormat::Alaw, AudioFormat::Mulaw, AudioFormat::ImaAdpcm].map(|format| Caps::Audio {
        format,
        channels: ANY_CHANNELS,
        sample_rate: ANY_SAMPLE_RATE,
    })
}

/// Read a `fmt ` chunk body. `None` when it is too short or names a payload this
/// does not model.
fn parse_fmt(body: &[u8]) -> Option<WaveFormat> {
    if body.len() < FMT_CHUNK_MIN_LEN {
        return None;
    }
    let mut tag = read_u16(body, 0)?;
    if tag == FORMAT_EXTENSIBLE {
        tag = read_u16(body, EXTENSIBLE_TAG_OFFSET)?;
    }
    let channels = read_u16(body, 2)?;
    let sample_rate = read_u32(body, 4)?;
    let bits = read_u16(body, 14)?;
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

    /// The formats a WAV file can carry, advertised at negotiation; the
    /// concrete one is fixed via `CapsChanged` once `fmt ` is read.
    fn output_alternatives() -> CapsSet {
        let mut alternatives = Vec::from(pcm_formats().map(|format| Caps::Audio {
            format,
            channels: ANY_CHANNELS,
            sample_rate: ANY_SAMPLE_RATE,
        }));
        alternatives.extend(coded_alternatives());
        CapsSet::from_alternatives(alternatives)
    }

    /// Read the RIFF header and the chunks ahead of `data`, then push everything
    /// buffered after it as samples.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if !self.riff_seen {
            if self.buf.len() < RIFF_HEADER_LEN {
                return Ok(());
            }
            if read_fourcc(&self.buf, 0) != Some(RIFF_FOURCC)
                || read_fourcc(&self.buf, CHUNK_HEADER_LEN) != Some(*b"WAVE")
            {
                return Err(G2gError::CapsMismatch);
            }
            self.buf.drain(..RIFF_HEADER_LEN);
            self.riff_seen = true;
        }
        while !self.in_data {
            if self.buf.len() < CHUNK_HEADER_LEN {
                return Ok(());
            }
            let id = read_fourcc(&self.buf, 0).ok_or(G2gError::CapsMismatch)?;
            let size = read_u32(&self.buf, FOURCC_LEN).ok_or(G2gError::CapsMismatch)? as usize;
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
            let padded = padded_len(size).ok_or(G2gError::CapsMismatch)?;
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

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    /// The layout is unknown until the `fmt ` chunk is read, so this advertises
    /// what the audio decoder does (M754): a concrete default rate first, the
    /// `ANY_SAMPLE_RATE` wildcard second so a downstream `rate=` pin intersects
    /// to it. A lone wildcard cannot fixate. The real rate, channel count and
    /// format arrive with the `CapsChanged` the parsed header emits. The coded
    /// payloads (M1073) come last, so a downstream that takes anything still
    /// gets PCM.
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
                let mut alternatives = alloc::vec![pcm(48_000), pcm(ANY_SAMPLE_RATE)];
                alternatives.extend(coded_alternatives());
                CapsSet::from_alternatives(alternatives)
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
