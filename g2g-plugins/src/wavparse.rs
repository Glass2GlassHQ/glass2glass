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
//! `data` is the last chunk this reads: its samples end at its declared size,
//! and run to end of stream for a size of `0xFFFFFFFF` (the streaming sentinel
//! `wavenc` writes). PCM leaves in whole sample frames, so a chunk boundary in
//! the input never splits a sample.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    pcm_formats, AsyncElement, AudioFormat, BusHandle, BusMessage, ByteStreamEncoding, Caps,
    CapsConstraint, CapsSet, ChannelLayout, ConfigureOutcome, ElementMetadata, FrameTiming,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, ANY_CHANNELS, ANY_SAMPLE_RATE,
};

use crate::codectag::{audio_from_stream_name, audio_stream_name};
use crate::riff::{
    chunks, padded_len, read_fourcc, read_u16, read_u32, CHUNK_HEADER_LEN, FOURCC_LEN, RIFF_FOURCC,
    RIFF_HEADER_LEN,
};
use crate::waveformat::{audio_format, sample_frame_len, tagged_audio};

/// The fields of a `fmt ` chunk this reads: tag, channels, rate, byte rate,
/// block align, bit depth.
const FMT_CHUNK_MIN_LEN: usize = 16;

/// WAVE_FORMAT_EXTENSIBLE: the real tag sits in the chunk's extension GUID, whose
/// first two bytes are the tag it extends.
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;
/// Offset of that GUID within a `fmt ` chunk (16 fixed bytes, then `cbSize`).
const EXTENSIBLE_TAG_OFFSET: usize = 24;
/// Offset of `dwChannelMask` in a WAVEFORMATEXTENSIBLE `fmt ` chunk: the 18-byte
/// WAVEFORMATEX, then the 2-byte `wValidBitsPerSample` union.
const EXTENSIBLE_CHANNEL_MASK_OFFSET: usize = 20;
/// A `fmt ` chunk short enough to lack the extension fields, so no channel mask.
const EXTENSIBLE_FMT_MIN_LEN: usize = 40;

const NANOS_PER_SECOND: u64 = 1_000_000_000;
/// The `data` chunk size a streaming writer stamps when the length is not known
/// ahead of time (what `wavenc` writes), so it is not a real length.
const UNKNOWN_DATA_LEN: usize = u32::MAX as usize;

/// The PCM stream a `fmt ` chunk describes.
#[derive(Debug, Clone, Copy, PartialEq)]
struct WaveFormat {
    format: AudioFormat,
    channels: u8,
    /// The speakers `dwChannelMask` names, unspecified unless the file is
    /// WAVE_FORMAT_EXTENSIBLE and the mask agrees with the channel count.
    channel_layout: ChannelLayout,
    sample_rate: u32,
    /// `nAvgBytesPerSec`, what the `data` chunk length divides by to give the
    /// stream's duration. `0` when the file declares none.
    byte_rate: u32,
    block_align: u16,
}

/// The coded payloads this reads beside PCM.
const CODED_FORMATS: [AudioFormat; 3] =
    [AudioFormat::Alaw, AudioFormat::Mulaw, AudioFormat::ImaAdpcm];

/// The coded payloads a WAV file can carry, at the unknown-layout sentinels a
/// decoder negotiates against before the `fmt ` chunk is read.
fn coded_alternatives() -> [Caps; 3] {
    CODED_FORMATS.map(coded_caps)
}

fn coded_caps(format: AudioFormat) -> Caps {
    Caps::Audio {
        format,
        channels: ANY_CHANNELS,
        sample_rate: ANY_SAMPLE_RATE,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

/// Read a `fmt ` chunk body. `None` when it is too short or names a payload this
/// does not model.
fn parse_fmt(body: &[u8]) -> Option<WaveFormat> {
    if body.len() < FMT_CHUNK_MIN_LEN {
        return None;
    }
    let tag = format_tag(body)?;
    let mut channel_layout = ChannelLayout::UNSPECIFIED;
    if read_u16(body, 0)? == FORMAT_EXTENSIBLE && body.len() >= EXTENSIBLE_FMT_MIN_LEN {
        let mask = read_u32(body, EXTENSIBLE_CHANNEL_MASK_OFFSET)?;
        channel_layout = ChannelLayout::from_mask(mask as u16);
    }
    let channels = read_u16(body, 2)?;
    let sample_rate = read_u32(body, 4)?;
    let byte_rate = read_u32(body, 8)?;
    let block_align = read_u16(body, BLOCK_ALIGN_OFFSET)?;
    let bits = read_u16(body, BITS_OFFSET)?;
    if channels == 0 || channels > u8::MAX as u16 || sample_rate == 0 {
        return None;
    }
    // A mask whose speaker count disagrees with nChannels (or that named only
    // positions this does not model) describes a different stream than the one
    // being read, so fall back to the count convention.
    if channel_layout.channels() != channels as u8 {
        channel_layout = ChannelLayout::UNSPECIFIED;
    }
    let format = match audio_format(tag, bits) {
        Some(_) => readable_format(tag, bits)?,
        None => tagged_audio(tag)?,
    };
    Some(WaveFormat {
        format,
        channels: channels as u8,
        channel_layout,
        sample_rate,
        byte_rate,
        block_align,
    })
}

const BLOCK_ALIGN_OFFSET: usize = 12;
const BITS_OFFSET: usize = 14;

// the chunk's `wFormatTag`, or for WAVE_FORMAT_EXTENSIBLE the tag its GUID extends
fn format_tag(body: &[u8]) -> Option<u16> {
    match read_u16(body, 0)? {
        FORMAT_EXTENSIBLE => read_u16(body, EXTENSIBLE_TAG_OFFSET),
        tag => Some(tag),
    }
}

fn readable_format(tag: u16, bits: u16) -> Option<AudioFormat> {
    audio_format(tag, bits)
        .filter(|format| pcm_formats().contains(format) || CODED_FORMATS.contains(format))
}

impl WaveFormat {
    fn caps(&self) -> Caps {
        Caps::Audio {
            format: self.format,
            channels: self.channels,
            sample_rate: self.sample_rate,
            channel_layout: self.channel_layout,
        }
    }
}

// what a WAV file carries: the caps `WavParse` emits for it, or a `wFormatTag` it does not read
#[derive(Debug, Clone, PartialEq)]
pub enum WavStream {
    Readable(Caps),
    UnreadableTag(u16),
}

// `None` for a prefix that is not RIFF/WAVE or ends before a whole `fmt ` chunk
pub fn probe_stream(prefix: &[u8]) -> Option<WavStream> {
    if read_fourcc(prefix, 0) != Some(RIFF_FOURCC)
        || read_fourcc(prefix, CHUNK_HEADER_LEN) != Some(*b"WAVE")
    {
        return None;
    }
    let fmt = chunks(prefix, RIFF_HEADER_LEN..prefix.len()).find(|chunk| &chunk.id == b"fmt ")?;
    let body = fmt.body(prefix);
    if let Some(format) = parse_fmt(body) {
        // a decoder's coded input pad takes only the unknown-layout sentinels
        let caps = if CODED_FORMATS.contains(&format.format)
            || matches!(format.format, AudioFormat::Tagged(_))
        {
            coded_caps(format.format)
        } else {
            format.caps()
        };
        return Some(WavStream::Readable(caps));
    }
    let tag = format_tag(body)?;
    let bits = read_u16(body, BITS_OFFSET)?;
    readable_format(tag, bits)
        .is_none()
        .then_some(WavStream::UnreadableTag(tag))
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
    // the codec known only by its tag that the output negotiates as, set by the `stream` property
    tagged: Option<AudioFormat>,
    // the `fmt ` body a tagged stream sends ahead of its data, for the decoder to open from
    tagged_header: Vec<u8>,
    bus: Option<BusHandle>,
    /// True once the `data` chunk header has been consumed: everything after it
    /// is samples.
    in_data: bool,
    // None when the data chunk size is the streaming sentinel
    data_remaining: Option<usize>,
    // bytes of the `data` chunk pushed so far, which time the next buffer
    data_offset: u64,
    emitted: u64,
}

impl WavParse {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach the pipeline bus so the stream's length, computed from the `data`
    /// chunk and the `fmt ` byte rate, posts as a
    /// [`BusMessage::DurationChanged`] once the `data` header is read. WAV is
    /// demuxed rather than sourced, so the runner's `query_duration` path never
    /// sees it.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
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
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }));
        alternatives.extend(coded_alternatives());
        CapsSet::from_alternatives(alternatives)
    }

    /// Post the stream's duration from the `data` chunk length and the byte
    /// rate. Both come from the file, so a zero rate or a streaming file's
    /// unknown-length sentinel reports nothing rather than a bogus number.
    fn post_duration(&self, data_len: usize, format: WaveFormat) {
        let Some(bus) = &self.bus else {
            return;
        };
        if format.byte_rate == 0 || data_len == 0 || data_len == UNKNOWN_DATA_LEN {
            return;
        }
        let duration_ns =
            (data_len as u64).saturating_mul(NANOS_PER_SECOND) / u64::from(format.byte_rate);
        if duration_ns > 0 {
            bus.try_post(BusMessage::DurationChanged { duration_ns });
        }
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
                if self.tagged.is_some_and(|tagged| tagged != format.format) {
                    return Err(G2gError::CapsMismatch);
                }
                self.post_duration(size, format);
                self.data_remaining = (size != UNKNOWN_DATA_LEN).then_some(size);
                out.push(PipelinePacket::CapsChanged(format.caps())).await?;
                if matches!(format.format, AudioFormat::Tagged(_)) {
                    let header = core::mem::take(&mut self.tagged_header);
                    self.push_frame(header, FrameTiming::default(), out).await?;
                }
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
                let body = &self.buf[CHUNK_HEADER_LEN..CHUNK_HEADER_LEN + size];
                self.format = Some(parse_fmt(body).ok_or(G2gError::CapsMismatch)?);
                self.tagged_header = body.to_vec();
            }
            self.buf.drain(..total);
        }
        let format = self.format.ok_or(G2gError::CapsMismatch)?;
        // Coded payloads have no fixed frame width, a tagged codec decodes whole blocks.
        let frame_len = match format.format {
            AudioFormat::Tagged(_) => usize::from(format.block_align.max(1)),
            _ => sample_frame_len(format.format, format.channels).unwrap_or(1),
        };
        if self
            .data_remaining
            .is_some_and(|remaining| remaining < frame_len)
        {
            // Past the end of the `data` chunk.
            self.buf.clear();
            return Ok(());
        }
        let available = self
            .data_remaining
            .map_or(self.buf.len(), |remaining| remaining.min(self.buf.len()));
        let whole_frames = available - available % frame_len;
        if whole_frames == 0 {
            return Ok(());
        }
        let rest = self.buf.split_off(whole_frames);
        let samples = core::mem::replace(&mut self.buf, rest);
        if let Some(remaining) = &mut self.data_remaining {
            *remaining -= whole_frames;
        }
        let start = self.data_offset;
        self.data_offset = start.saturating_add(whole_frames as u64);
        let timing = data_timing(start, self.data_offset, format.byte_rate);
        self.push_frame(samples, timing, out).await
    }

    async fn push_frame(
        &mut self,
        bytes: Vec<u8>,
        timing: FrameTiming,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing,
            self.emitted,
        );
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}

// The `data` bytes from `start` to `end` at the header's byte rate, rounded up as gst's wavparse does.
fn data_timing(start: u64, end: u64, byte_rate: u32) -> FrameTiming {
    if byte_rate == 0 {
        return FrameTiming {
            pts_ns: FrameTiming::PTS_NONE,
            ..FrameTiming::default()
        };
    }
    let offset_ns = |offset: u64| -> u64 {
        (u128::from(offset) * u128::from(NANOS_PER_SECOND))
            .div_ceil(u128::from(byte_rate))
            .try_into()
            .unwrap_or(FrameTiming::PTS_NONE)
    };
    let pts_ns = offset_ns(start);
    FrameTiming {
        pts_ns,
        dts_ns: pts_ns,
        duration_ns: offset_ns(end).saturating_sub(pts_ns),
        ..FrameTiming::default()
    }
}

static WAVPARSE_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "stream",
    PropKind::Str,
    "the format the output negotiates as: auto (PCM or the coded payloads the parser reads) | audio-riff-0x<tag> (a codec known only by its tag, with ffmpeg-long-tail)",
)
.with_default("auto")];

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
        let tagged = self.tagged;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Wav,
            } if tagged.is_some() => {
                CapsSet::from_alternatives(tagged.map(coded_caps).into_iter().collect())
            }
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Wav,
            } => {
                let pcm = |sample_rate| Caps::Audio {
                    format: AudioFormat::PcmS16Le,
                    channels: ANY_CHANNELS,
                    sample_rate,
                    channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
                };
                let mut alternatives = alloc::vec![pcm(48_000), pcm(ANY_SAMPLE_RATE)];
                alternatives.extend(coded_alternatives());
                CapsSet::from_alternatives(alternatives)
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn properties(&self) -> &'static [PropertySpec] {
        WAVPARSE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "stream" => {
                let value = value.as_str().ok_or(PropError::Type)?;
                self.tagged = match value {
                    "auto" => None,
                    name => Some(audio_from_stream_name(&[], name).ok_or(PropError::Value)?),
                };
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "stream" => Some(PropValue::Str(match self.tagged {
                Some(format) => audio_stream_name(&[], format)?,
                None => String::from("auto"),
            })),
            _ => None,
        }
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
