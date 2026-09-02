//! Sun / NeXT AU (`.au` / `.snd`): `auparse` reads the 24-byte header plus the
//! samples it describes, `avmux_au` writes one. The Unix sibling of WAVE.
//!
//! ```text
//! filesrc location=in.au ! auparse ! audioconvert ! autoaudiosink
//! audiotestsrc ! avmux_au ! filesink location=out.au
//! ```
//!
//! The header is six big-endian `u32`s: `.snd` magic, the byte offset of the
//! samples (at least 24, and it may skip an annotation), a data size
//! (`0xFFFFFFFF` when unknown), an encoding tag, the sample rate, and the
//! channel count. Everything is attacker-controlled: a header claiming a
//! terabyte offset or a nonsense encoding fails the parse rather than
//! allocating or looping.
//!
//! Multi-byte PCM is big-endian on the wire and 8-bit linear samples are signed.
//! Both are converted at this boundary so the rest of the graph sees the
//! little-endian / unsigned `AudioFormat`s. The muxer writes the streaming
//! sentinel in the data-size field.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

use crate::audiocontainer::{
    container_caps, emit_pcm, mux_constraint, mux_input_alternatives, parse_constraint,
    parse_output_alternatives, PcmContainer, PcmMux, MAX_CHANNELS, MAX_SAMPLE_RATE, STREAMING_SIZE,
};
use crate::pcmendian::{read_u32_be, PcmWire};

/// The six-word header every AU file opens with.
const HEADER_LEN: usize = 24;
/// Encoding tag: 8-bit mu-law.
const ENC_MULAW: u32 = 1;
/// Encoding tag: 8-bit signed linear.
const ENC_S8: u32 = 2;
/// Encoding tag: 16-bit signed linear, big-endian.
const ENC_S16: u32 = 3;
/// Encoding tag: 24-bit signed linear, big-endian.
const ENC_S24: u32 = 4;
/// Encoding tag: 32-bit signed linear, big-endian.
const ENC_S32: u32 = 5;
/// Encoding tag: 32-bit IEEE float, big-endian.
const ENC_F32: u32 = 6;
/// Encoding tag: 8-bit A-law.
const ENC_ALAW: u32 = 27;
/// Longest annotation (the gap between the 24-byte header and the samples) this
/// will skip. A header claiming more is treated as corrupt.
const MAX_ANNOTATION: usize = 1 << 16;

const MAGIC: &[u8; 4] = b".snd";

fn au_caps() -> Caps {
    container_caps(ByteStreamEncoding::Au)
}

/// The header is written with the streaming size sentinel, so nothing is held
/// back and the samples need no padding.
static AU_CONTAINER: PcmContainer = PcmContainer {
    encoding: ByteStreamEncoding::Au,
    supported,
    header,
    finalize_at_eos: false,
    data_alignment: 1,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AuFormat {
    wire: PcmWire,
    /// Bytes of annotation still to skip after the 24-byte header.
    data_offset: usize,
    data_remaining: Option<usize>,
}

fn encoding_tag(format: AudioFormat) -> Option<u32> {
    Some(match format {
        AudioFormat::Mulaw => ENC_MULAW,
        AudioFormat::PcmU8 => ENC_S8,
        AudioFormat::PcmS16Le => ENC_S16,
        AudioFormat::PcmS24Le => ENC_S24,
        AudioFormat::PcmS32Le => ENC_S32,
        AudioFormat::PcmF32Le => ENC_F32,
        AudioFormat::Alaw => ENC_ALAW,
        _ => return None,
    })
}

fn format_from_tag(tag: u32) -> Option<AudioFormat> {
    Some(match tag {
        ENC_MULAW => AudioFormat::Mulaw,
        ENC_S8 => AudioFormat::PcmU8,
        ENC_S16 => AudioFormat::PcmS16Le,
        ENC_S24 => AudioFormat::PcmS24Le,
        ENC_S32 => AudioFormat::PcmS32Le,
        ENC_F32 => AudioFormat::PcmF32Le,
        ENC_ALAW => AudioFormat::Alaw,
        _ => return None,
    })
}

fn parse_header(bytes: &[u8]) -> Option<AuFormat> {
    if bytes.len() < HEADER_LEN || &bytes[..4] != MAGIC {
        return None;
    }
    let data_offset = read_u32_be(bytes, 4)? as usize;
    if data_offset < HEADER_LEN || data_offset.saturating_sub(HEADER_LEN) > MAX_ANNOTATION {
        return None;
    }
    let tag = read_u32_be(bytes, 12)?;
    let data_size = read_u32_be(bytes, 8)?;
    let sample_rate = read_u32_be(bytes, 16)?;
    let channels = read_u32_be(bytes, 20)?;
    if sample_rate == 0
        || sample_rate > MAX_SAMPLE_RATE
        || channels == 0
        || channels > u32::from(MAX_CHANNELS)
    {
        return None;
    }
    let format = format_from_tag(tag)?;
    Some(AuFormat {
        wire: PcmWire {
            format,
            channels: channels as u8,
            sample_rate,
            big_endian: true,
        },
        data_offset: data_offset - HEADER_LEN,
        data_remaining: (data_size != STREAMING_SIZE).then_some(data_size as usize),
    })
}

fn header(
    format: AudioFormat,
    channels: u16,
    sample_rate: u32,
    _payload_size: usize,
) -> Option<Vec<u8>> {
    let tag = encoding_tag(format)?;
    let mut out = Vec::with_capacity(HEADER_LEN);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(HEADER_LEN as u32).to_be_bytes());
    out.extend_from_slice(&STREAMING_SIZE.to_be_bytes());
    out.extend_from_slice(&tag.to_be_bytes());
    out.extend_from_slice(&sample_rate.to_be_bytes());
    out.extend_from_slice(&(channels as u32).to_be_bytes());
    Some(out)
}

/// Parses a Sun / NeXT AU byte stream into the PCM stream it carries.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::au::AuParse;
///
/// let parse = AuParse::new();
/// ```
#[derive(Debug, Default)]
pub struct AuParse {
    configured: bool,
    buf: Vec<u8>,
    format: Option<AuFormat>,
    header_seen: bool,
    emitted: u64,
}

impl AuParse {
    pub fn new() -> Self {
        Self::default()
    }

    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if !self.header_seen {
            if self.buf.len() < HEADER_LEN {
                return Ok(());
            }
            let parsed = parse_header(&self.buf).ok_or(G2gError::CapsMismatch)?;
            self.buf.drain(..HEADER_LEN);
            out.push(PipelinePacket::CapsChanged(Caps::Audio {
                format: parsed.wire.format,
                channels: parsed.wire.channels,
                sample_rate: parsed.wire.sample_rate,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            }))
            .await?;
            self.format = Some(parsed);
            self.header_seen = true;
        }
        let parsed = self.format.ok_or(G2gError::NotConfigured)?;
        if parsed.data_offset > 0 {
            if self.buf.len() < parsed.data_offset {
                return Ok(());
            }
            self.buf.drain(..parsed.data_offset);
            if let Some(stored) = self.format.as_mut() {
                stored.data_offset = 0;
            }
        }
        let stored = self.format.as_mut().ok_or(G2gError::NotConfigured)?;
        emit_pcm(
            &mut self.buf,
            &mut stored.data_remaining,
            &parsed.wire,
            &mut self.emitted,
            out,
        )
        .await
    }
}

impl AsyncElement for AuParse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "AU parser",
            "Codec/Demuxer/Audio",
            "Reads the PCM stream out of a Sun / NeXT AU byte stream",
            "g2g",
        )
    }

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&au_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        parse_constraint(ByteStreamEncoding::Au)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Au
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
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for AuParse {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(au_caps())),
            PadTemplate::source(parse_output_alternatives()),
        ])
    }
}

/// Writes a PCM stream as a Sun / NeXT AU byte stream.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::au::AuMux;
///
/// let element = AuMux::new();
/// ```
#[derive(Debug, Default)]
pub struct AuMux {
    inner: PcmMux,
}

impl AuMux {
    pub fn new() -> Self {
        Self::default()
    }
}

fn supported(format: AudioFormat) -> bool {
    encoding_tag(format).is_some()
}

impl AsyncElement for AuMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn is_format_boundary(&self) -> bool {
        true
    }

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        PcmMux::intercept(upstream_caps, &AU_CONTAINER)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        mux_constraint(&AU_CONTAINER)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.inner.configure(absolute_caps, &AU_CONTAINER)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        self.inner.process(packet, out, &AU_CONTAINER)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "AU encoder",
            "Codec/Muxer/Audio",
            "Wraps raw PCM in a Sun / NeXT AU byte stream",
            "g2g",
        )
    }
}

impl PadTemplates for AuMux {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(mux_input_alternatives())),
            PadTemplate::source(CapsSet::one(au_caps())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcmendian::{convert_wire_layout, sample_width, CARRIED};
    use crate::testutil::{data_bytes, roundtrip, run};
    use crate::typefind::sniff;
    use alloc::vec::Vec;

    fn audio(format: AudioFormat, channels: u8, rate: u32) -> Caps {
        Caps::Audio {
            format,
            channels,
            sample_rate: rate,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    fn samples(format: AudioFormat, channels: u8, frames: usize) -> Vec<u8> {
        let n = sample_width(format).unwrap() * channels as usize * frames;
        (0..n).map(|i| i as u8).collect()
    }

    #[test]
    fn header_and_parse_agree_for_every_carried_format() {
        for format in CARRIED {
            let channels = 2u16;
            let rate = 48_000u32;
            let bytes = header(format, channels, rate, 0).unwrap();
            assert_eq!(&bytes[..4], MAGIC);
            let parsed = parse_header(&bytes).unwrap();
            assert_eq!(parsed.wire.format, format);
            assert_eq!(parsed.wire.channels, channels as u8);
            assert_eq!(parsed.wire.sample_rate, rate);
            assert_eq!(parsed.data_offset, 0);
            assert_eq!(parsed.data_remaining, None);
        }
    }

    #[test]
    fn every_carried_format_round_trips() {
        for format in CARRIED {
            for channels in [1u8, 2] {
                roundtrip(
                    AuMux::new(),
                    AuParse::new(),
                    audio(format, channels, 8_000),
                    au_caps(),
                    &samples(format, channels, 8),
                );
            }
        }
    }

    #[test]
    fn a_written_file_types_as_au() {
        let out = run(
            &mut AuMux::new(),
            &audio(AudioFormat::PcmS16Le, 1, 8_000),
            &[&samples(AudioFormat::PcmS16Le, 1, 4)],
        );
        assert_eq!(
            sniff(&data_bytes(&out.packets)),
            Some(ByteStreamEncoding::Au)
        );
    }

    #[test]
    fn a_truncated_or_unknown_header_is_rejected() {
        assert!(parse_header(b".snd").is_none());
        let mut bad = header(AudioFormat::PcmS16Le, 1, 8_000, 0).unwrap();
        let tag_at = 12;
        bad[tag_at..tag_at + 4].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(parse_header(&bad).is_none());
    }

    #[test]
    fn declared_data_size_excludes_trailing_bytes() {
        let pcm = samples(AudioFormat::PcmS16Le, 1, 4);
        let mut wire_pcm = pcm.clone();
        convert_wire_layout(&mut wire_pcm, AudioFormat::PcmS16Le, true);
        let mut file = header(AudioFormat::PcmS16Le, 1, 8_000, 0).unwrap();
        file[8..12].copy_from_slice(&(wire_pcm.len() as u32).to_be_bytes());
        file.extend_from_slice(&wire_pcm);
        file.extend_from_slice(b"not audio");

        let parsed = run(&mut AuParse::new(), &au_caps(), &[&file]);
        assert_eq!(data_bytes(&parsed.packets), pcm);
    }
}
