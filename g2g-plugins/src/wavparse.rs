// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! WAV parser: a RIFF/WAVE byte stream in (`Caps::ByteStream{Wav}`), the audio
//! it carries out (`Caps::Audio`), so `filesrc location=x.wav ! decodebin`
//! reaches an audio element. The read side of [`crate::wavenc::WavEnc`]. A
//! G.711 or IMA ADPCM payload comes out coded, for [`crate::g711`] /
//! [`crate::adpcm`] to decode, and MP3 or AC-3 for its parser and decoder.
//!
//! RIFF is a chunk list: a 12-byte `RIFF....WAVE` header, then `id` + `size`
//! chunks. Only `fmt ` (the sample format), `fact` (a coded payload's sample
//! count) and `data` (the samples) matter here; anything else (`LIST`, a
//! writer's own metadata) is skipped by its
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
    PropKind, PropValue, PropertySpec, TagList, ANY_CHANNELS, ANY_SAMPLE_RATE,
};

use crate::codectag::{audio_from_stream_name, audio_stream_name};
use crate::id3::{parse_id3v2, MAX_ID3V2_TAG_PARSED};
use crate::riff::{
    chunks, padded_len, parse_info, read_fourcc, read_u16, read_u32, read_u64, FourCc,
    CHUNK_HEADER_LEN, FOURCC_LEN, INFO_FOURCC, LIST_FOURCC, RIFF_FOURCC, RIFF_HEADER_LEN,
};
use crate::waveformat::{audio_format, sample_frame_len, tagged_audio};
use g2g_mcu::adpcm::samples_per_block_channels;

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
const CODED_FORMATS: [AudioFormat; 5] = [
    AudioFormat::Alaw,
    AudioFormat::Mulaw,
    AudioFormat::ImaAdpcm,
    AudioFormat::Mp3,
    AudioFormat::Ac3,
];

/// The coded payloads a WAV file can carry, at the unknown-layout sentinels a
/// decoder negotiates against before the `fmt ` chunk is read.
fn coded_alternatives() -> [Caps; 5] {
    CODED_FORMATS.map(coded_caps)
}

// `stream=` names for the payloads the solve cannot tell apart, since their parsers pass any caps
const CODED_STREAM_NAMES: &[(&str, AudioFormat)] =
    &[("mp3", AudioFormat::Mp3), ("ac3", AudioFormat::Ac3)];

pub(crate) fn audio_stream_str(format: AudioFormat) -> Option<String> {
    audio_stream_name(CODED_STREAM_NAMES, format)
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
    let declared_block_align = read_u16(body, BLOCK_ALIGN_OFFSET)?;
    // ffmpeg's wav demuxer reads XMA in whole packets, whatever block align the header gives
    let block_align = match tag {
        WAVE_FORMAT_XMA1 | WAVE_FORMAT_XMA2 => XMA_PACKET_LEN,
        // ffmpeg's wav demuxer reads MS ADPCM past stereo as one declared block per channel
        WAVE_FORMAT_MS_ADPCM if channels > 2 => declared_block_align.checked_mul(channels)?,
        _ => declared_block_align,
    };
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
    // some writers put a wrong nAvgBytesPerSec in valid PCM and ADPCM files
    let byte_rate = if let Some(frame_len) = sample_frame_len(format, channels as u8) {
        u32::try_from(frame_len).ok()?.checked_mul(sample_rate)?
    } else if let Some(block_samples) = adpcm_block_samples(format, tag, block_align, channels) {
        let rate = u64::from(sample_rate) * u64::from(block_align) / block_samples;
        u32::try_from(rate).ok()?
    } else {
        byte_rate
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

// samples one channel carries in an ADPCM block, `None` for other payloads
fn adpcm_block_samples(
    format: AudioFormat,
    tag: u16,
    block_align: u16,
    channels: u16,
) -> Option<u64> {
    if format == AudioFormat::ImaAdpcm {
        let samples = samples_per_block_channels(usize::from(block_align), usize::from(channels));
        return Some(samples as u64);
    }
    if tag != WAVE_FORMAT_MS_ADPCM {
        return None;
    }
    // MS ADPCM opens each channel with a 7-byte header holding two samples, then packs two samples per byte
    const MS_ADPCM_CHANNEL_HEADER_LEN: u16 = 7;
    let data_bytes = block_align.checked_sub(MS_ADPCM_CHANNEL_HEADER_LEN.checked_mul(channels)?)?;
    Some(u64::from(data_bytes) * 2 / u64::from(channels) + 2)
}

const BLOCK_ALIGN_OFFSET: usize = 12;
// WAVEFORMATEX is 18 bytes followed by at most a 16-bit cbSize of extra bytes
const MAX_FMT_CHUNK_LEN: usize = 18 + u16::MAX as usize;
const BITS_OFFSET: usize = 14;
const WAVE_FORMAT_MS_ADPCM: u16 = 0x0002;
const WAVE_FORMAT_XMA1: u16 = 0x0165;
const WAVE_FORMAT_XMA2: u16 = 0x0166;
const XMA_PACKET_LEN: u16 = 2048;

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

// RF64 (EBU Tech 3306) and BW64 (ITU-R BS.2088) are WAVE files whose 64-bit sizes sit in a `ds64` chunk
const WAVE_FILE_IDS: [FourCc; 3] = [RIFF_FOURCC, *b"RF64", *b"BW64"];

pub(crate) fn opens_wave_file(prefix: &[u8]) -> bool {
    read_fourcc(prefix, 0).is_some_and(|id| WAVE_FILE_IDS.contains(&id))
        && read_fourcc(prefix, CHUNK_HEADER_LEN) == Some(*b"WAVE")
}

// `None` for a prefix that is not RIFF/WAVE or ends before a whole `fmt ` chunk
pub fn probe_stream(prefix: &[u8]) -> Option<WavStream> {
    if !opens_wave_file(prefix) {
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
    // the coded format the output negotiates as, set by the `stream` property
    stream_format: Option<AudioFormat>,
    // the `fmt ` body a tagged stream sends ahead of its data, for the decoder to open from
    tagged_header: Vec<u8>,
    bus: Option<BusHandle>,
    /// True once the `data` chunk header has been consumed: everything after it
    /// is samples.
    in_data: bool,
    // None when the data chunk size is the streaming sentinel
    data_remaining: Option<usize>,
    chunk_bytes_to_skip: usize,
    // the `fact` chunk's dwSampleLength, the sample count of a coded payload
    fact_samples: Option<u32>,
    // an RF64 / BW64 file's `ds64` data size
    ds64_data_len: Option<u64>,
    // the `data` chunk has ended, the chunks after it are read for tags
    data_done: bool,
    // the pad byte an odd-sized `data` chunk ends on
    data_pad: usize,
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
    /// rate, or for a coded payload from its `fact` sample count. All come from
    /// the file, so a zero rate or a streaming file's unknown-length sentinel
    /// reports nothing rather than a bogus number.
    fn post_duration(&self, data_len: usize, format: WaveFormat) {
        let Some(bus) = &self.bus else {
            return;
        };
        // a VBR MP3's nAvgBytesPerSec is only an average
        let fact_samples = self
            .fact_samples
            .filter(|_| sample_frame_len(format.format, format.channels).is_none());
        let duration_ns = if let Some(samples) = fact_samples {
            u64::from(samples).saturating_mul(NANOS_PER_SECOND) / u64::from(format.sample_rate)
        } else if format.byte_rate == 0 || data_len == 0 || data_len == UNKNOWN_DATA_LEN {
            return;
        } else {
            (data_len as u64).saturating_mul(NANOS_PER_SECOND) / u64::from(format.byte_rate)
        };
        if duration_ns > 0 {
            bus.try_post(BusMessage::DurationChanged { duration_ns });
        }
    }

    fn post_tags(&self, tags: TagList) {
        if let (Some(bus), false) = (&self.bus, tags.is_empty()) {
            bus.try_post(BusMessage::Tag {
                tags,
                program: None,
            });
        }
    }

    /// Read the RIFF header and the chunks ahead of `data`, then push everything
    /// buffered after it as samples.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if !self.riff_seen {
            if self.buf.len() < RIFF_HEADER_LEN {
                return Ok(());
            }
            if !opens_wave_file(&self.buf) {
                return Err(G2gError::CapsMismatch);
            }
            self.buf.drain(..RIFF_HEADER_LEN);
            self.riff_seen = true;
        }
        loop {
            self.walk_chunks(out).await?;
            if !self.in_data {
                return Ok(());
            }
            let format = self.format.ok_or(G2gError::CapsMismatch)?;
            // Coded payloads have no fixed frame width, a tagged codec decodes whole blocks.
            let frame_len = match format.format {
                AudioFormat::Tagged(_) => usize::from(format.block_align.max(1)),
                _ => sample_frame_len(format.format, format.channels).unwrap_or(1),
            };
            if let Some(remaining) = self
                .data_remaining
                .filter(|remaining| *remaining < frame_len)
            {
                // past the end of the `data` chunk, the chunks after it can hold tags
                self.in_data = false;
                self.data_done = true;
                self.chunk_bytes_to_skip = remaining + self.data_pad;
                continue;
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
            self.push_frame(samples, timing, out).await?;
        }
    }

    // Read the chunks ahead of the next sample bytes, up to the `data` header or the end of what has arrived.
    async fn walk_chunks(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        while !self.in_data {
            let skipped = self.chunk_bytes_to_skip.min(self.buf.len());
            self.buf.drain(..skipped);
            self.chunk_bytes_to_skip -= skipped;
            if self.chunk_bytes_to_skip > 0 || self.buf.len() < CHUNK_HEADER_LEN {
                return Ok(());
            }
            let id = read_fourcc(&self.buf, 0).ok_or(G2gError::CapsMismatch)?;
            let size = read_u32(&self.buf, FOURCC_LEN).ok_or(G2gError::CapsMismatch)? as usize;
            // A chunk body is padded to an even length, and the size comes from
            // the file: a chunk claiming more than the stream holds waits rather
            // than indexing past the buffer.
            let padded = padded_len(size).ok_or(G2gError::CapsMismatch)?;
            let is_tag_chunk = matches!(&id, b"LIST" | b"id3 " | b"ID3 ");
            if is_tag_chunk && size <= MAX_ID3V2_TAG_PARSED {
                let total = padded
                    .checked_add(CHUNK_HEADER_LEN)
                    .ok_or(G2gError::CapsMismatch)?;
                if self.buf.len() < total {
                    return Ok(());
                }
                let tags = match chunks(&self.buf, 0..total).next() {
                    Some(list) if list.id == LIST_FOURCC => match list.list_form(&self.buf) {
                        Some(INFO_FOURCC) => parse_info(&self.buf, list.list_body()),
                        _ => TagList::new(),
                    },
                    Some(id3) => parse_id3v2(id3.body(&self.buf)),
                    None => TagList::new(),
                };
                self.post_tags(tags);
                self.buf.drain(..total);
                continue;
            }
            if self.data_done {
                self.buf.drain(..CHUNK_HEADER_LEN);
                self.chunk_bytes_to_skip = padded;
                continue;
            }
            if &id == b"data" {
                self.buf.drain(..CHUNK_HEADER_LEN);
                self.data_pad = size % 2;
                let format = self.format.ok_or(G2gError::CapsMismatch)?;
                if self
                    .stream_format
                    .is_some_and(|stream_format| stream_format != format.format)
                {
                    return Err(G2gError::CapsMismatch);
                }
                let size = match (size, self.ds64_data_len) {
                    (UNKNOWN_DATA_LEN, Some(len)) => {
                        usize::try_from(len).map_err(|_| G2gError::CapsMismatch)?
                    }
                    _ => size,
                };
                self.post_duration(size, format);
                // gst wavparse reads a zero-length data chunk to the end of the file
                self.data_remaining = (size != UNKNOWN_DATA_LEN && size != 0).then_some(size);
                out.push(PipelinePacket::CapsChanged(format.caps())).await?;
                if matches!(format.format, AudioFormat::Tagged(_)) {
                    let header = core::mem::take(&mut self.tagged_header);
                    self.push_frame(header, FrameTiming::default(), out).await?;
                }
                self.in_data = true;
                break;
            }
            if &id == b"ds64" {
                // riffSize, then the 64-bit dataSize a `data` chunk sized 0xFFFFFFFF stands for
                const DATA_SIZE_AT: usize = CHUNK_HEADER_LEN + 8;
                const DATA_SIZE_LEN: usize = 8;
                if size >= DATA_SIZE_AT - CHUNK_HEADER_LEN + DATA_SIZE_LEN {
                    if self.buf.len() < DATA_SIZE_AT + DATA_SIZE_LEN {
                        return Ok(());
                    }
                    self.ds64_data_len = read_u64(&self.buf, DATA_SIZE_AT);
                }
                self.buf.drain(..CHUNK_HEADER_LEN);
                self.chunk_bytes_to_skip = padded;
                continue;
            }
            if &id == b"fact" {
                const SAMPLE_LENGTH_LEN: usize = 4;
                if size >= SAMPLE_LENGTH_LEN {
                    if self.buf.len() < CHUNK_HEADER_LEN + SAMPLE_LENGTH_LEN {
                        return Ok(());
                    }
                    self.fact_samples = read_u32(&self.buf, CHUNK_HEADER_LEN);
                }
                self.buf.drain(..CHUNK_HEADER_LEN);
                self.chunk_bytes_to_skip = padded;
                continue;
            }
            if &id != b"fmt " {
                self.buf.drain(..CHUNK_HEADER_LEN);
                self.chunk_bytes_to_skip = padded;
                continue;
            }
            if size > MAX_FMT_CHUNK_LEN {
                return Err(G2gError::CapsMismatch);
            }
            let total = padded
                .checked_add(CHUNK_HEADER_LEN)
                .ok_or(G2gError::CapsMismatch)?;
            if self.buf.len() < total {
                return Ok(());
            }
            let body = &self.buf[CHUNK_HEADER_LEN..CHUNK_HEADER_LEN + size];
            let format = parse_fmt(body).ok_or(G2gError::CapsMismatch)?;
            self.format = Some(format);
            // the decoder opens from this header, so it carries the block align the data is cut at
            self.tagged_header = body.to_vec();
            self.tagged_header[BLOCK_ALIGN_OFFSET..BITS_OFFSET]
                .copy_from_slice(&format.block_align.to_le_bytes());
            self.buf.drain(..total);
        }
        Ok(())
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
    "the format the output negotiates as: auto (PCM or the coded payloads the parser reads) | mp3 | ac3 | audio-riff-0x<tag> (a codec known only by its tag, with ffmpeg-long-tail)",
)
.with_default("auto")];

impl AsyncElement for WavParse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn set_bus(&mut self, bus: BusHandle) {
        self.bus = Some(bus);
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "WAV parser",
            "Codec/Demuxer/Audio",
            "Reads the PCM stream out of a RIFF/WAVE byte stream",
            crate::AARON_BOXER,
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
    /// what the audio decoder does: a concrete default rate first, the
    /// `ANY_SAMPLE_RATE` wildcard second so a downstream `rate=` pin intersects
    /// to it. A lone wildcard cannot fixate. The real rate, channel count and
    /// format arrive with the `CapsChanged` the parsed header emits. The coded
    /// payloads come last, so a downstream that takes anything still
    /// gets PCM.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let stream_format = self.stream_format;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Wav,
            } if stream_format.is_some() => {
                CapsSet::from_alternatives(stream_format.map(coded_caps).into_iter().collect())
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
                self.stream_format = match value {
                    "auto" => None,
                    name => Some(
                        audio_from_stream_name(CODED_STREAM_NAMES, name).ok_or(PropError::Value)?,
                    ),
                };
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "stream" => Some(PropValue::Str(match self.stream_format {
                Some(format) => audio_stream_str(format)?,
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
                    // gst wavparse fails a file that ends before its `data` chunk
                    if !self.in_data && !self.data_done {
                        return Err(G2gError::CapsMismatch);
                    }
                }
                // The byte stream carries no sample format; the concrete caps
                // come from the parsed `fmt ` chunk instead.
                PipelinePacket::CapsChanged(_) => {}
                // every byte stream opens with a Segment, so one after a RIFF header starts the next file, as `concat` sends
                PipelinePacket::Segment(segment) => {
                    if self.riff_seen {
                        *self = WavParse {
                            configured: self.configured,
                            stream_format: self.stream_format,
                            bus: self.bus.take(),
                            emitted: self.emitted,
                            ..WavParse::default()
                        };
                    }
                    out.push(PipelinePacket::Segment(segment)).await?;
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{frame, CollectSink};
    use g2g_core::ByteStreamEncoding;

    const CHANNELS: u16 = 2;
    const SAMPLE_RATE: u32 = 44_100;
    const BITS: u16 = 16;
    const FRAME_LEN: u32 = CHANNELS as u32 * BITS as u32 / 8;
    const BYTE_RATE: u32 = FRAME_LEN * SAMPLE_RATE;

    fn chunk(id: &[u8; 4], size: u32, body: &[u8]) -> Vec<u8> {
        let mut out = id.to_vec();
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(body);
        out
    }

    fn pcm_fmt(byte_rate: u32) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&1u16.to_le_bytes());
        body.extend_from_slice(&CHANNELS.to_le_bytes());
        body.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
        body.extend_from_slice(&byte_rate.to_le_bytes());
        body.extend_from_slice(&(FRAME_LEN as u16).to_le_bytes());
        body.extend_from_slice(&BITS.to_le_bytes());
        chunk(b"fmt ", body.len() as u32, &body)
    }

    fn riff_header() -> Vec<u8> {
        let mut out = chunk(b"RIFF", u32::MAX, b"WAVE");
        out.truncate(RIFF_HEADER_LEN);
        out
    }

    fn parser() -> WavParse {
        let mut parse = WavParse::new();
        parse
            .configure_pipeline(&Caps::ByteStream {
                encoding: ByteStreamEncoding::Wav,
            })
            .unwrap();
        parse
    }

    async fn push(parse: &mut WavParse, bytes: Vec<u8>, out: &mut CollectSink) {
        parse
            .process(PipelinePacket::DataFrame(frame(bytes)), out)
            .await
            .unwrap();
    }

    fn data_frames(out: &CollectSink) -> Vec<(u64, usize)> {
        out.packets
            .iter()
            .filter_map(|packet| match packet {
                PipelinePacket::DataFrame(frame) => Some((
                    frame.timing.pts_ns,
                    frame.domain.as_system_slice().unwrap().len(),
                )),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn a_segment_after_a_file_starts_the_next_file() {
        const SAMPLES_BYTES: usize = 4 * FRAME_LEN as usize;
        let mut file = riff_header();
        file.extend(pcm_fmt(BYTE_RATE));
        file.extend(chunk(b"data", SAMPLES_BYTES as u32, &[0; SAMPLES_BYTES]));
        let mut parse = parser();
        let mut out = CollectSink::default();
        push(&mut parse, file.clone(), &mut out).await;
        parse
            .process(PipelinePacket::Segment(g2g_core::Segment::new()), &mut out)
            .await
            .unwrap();
        push(&mut parse, file, &mut out).await;

        assert_eq!(data_frames(&out), [(0, SAMPLES_BYTES), (0, SAMPLES_BYTES)]);
    }

    #[tokio::test]
    async fn pcm_is_timed_by_its_frame_size_not_the_declared_byte_rate() {
        const SAMPLES_BYTES: usize = 4 * FRAME_LEN as usize;
        let mut parse = parser();
        let mut out = CollectSink::default();
        let mut head = riff_header();
        head.extend(pcm_fmt(BYTE_RATE / 2));
        head.extend(chunk(
            b"data",
            2 * SAMPLES_BYTES as u32,
            &[0; SAMPLES_BYTES],
        ));
        push(&mut parse, head, &mut out).await;
        push(&mut parse, alloc::vec![0; SAMPLES_BYTES], &mut out).await;

        let expected_pts = (SAMPLES_BYTES as u64 * NANOS_PER_SECOND).div_ceil(u64::from(BYTE_RATE));
        assert_eq!(
            data_frames(&out),
            [(0, SAMPLES_BYTES), (expected_pts, SAMPLES_BYTES)]
        );
    }

    #[tokio::test]
    async fn ima_adpcm_is_timed_by_its_block_layout_not_the_declared_byte_rate() {
        use crate::waveformat::{IMA_ADPCM_BITS, WAVE_FORMAT_IMA_ADPCM};
        const BLOCK_ALIGN: u16 = 256;
        // mono IMA ADPCM: the 4-byte block header holds one sample, each of the other 252 bytes two
        const BLOCK_SAMPLES: u64 = 505;
        const ADPCM_SAMPLE_RATE: u32 = 10_100;
        const WRONG_BYTE_RATE: u32 = 4_000;
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&WAVE_FORMAT_IMA_ADPCM.to_le_bytes());
        fmt.extend_from_slice(&1u16.to_le_bytes());
        fmt.extend_from_slice(&ADPCM_SAMPLE_RATE.to_le_bytes());
        fmt.extend_from_slice(&WRONG_BYTE_RATE.to_le_bytes());
        fmt.extend_from_slice(&BLOCK_ALIGN.to_le_bytes());
        fmt.extend_from_slice(&IMA_ADPCM_BITS.to_le_bytes());
        let block = [0; BLOCK_ALIGN as usize];
        let mut head = riff_header();
        head.extend(chunk(b"fmt ", fmt.len() as u32, &fmt));
        head.extend(chunk(b"data", 2 * u32::from(BLOCK_ALIGN), &block));
        let mut parse = parser();
        let mut out = CollectSink::default();
        push(&mut parse, head, &mut out).await;
        push(&mut parse, block.to_vec(), &mut out).await;

        let block_ns = BLOCK_SAMPLES * NANOS_PER_SECOND / u64::from(ADPCM_SAMPLE_RATE);
        assert_eq!(
            data_frames(&out),
            [(0, block.len()), (block_ns, block.len())]
        );
    }

    #[tokio::test]
    async fn a_coded_payload_takes_its_duration_from_the_fact_sample_count() {
        const FACT_SAMPLES: u32 = 2 * SAMPLE_RATE;
        const AVERAGE_BYTE_RATE: u32 = 16_000;
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&crate::waveformat::WAVE_FORMAT_MP3.to_le_bytes());
        fmt.extend_from_slice(&CHANNELS.to_le_bytes());
        fmt.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
        fmt.extend_from_slice(&AVERAGE_BYTE_RATE.to_le_bytes());
        fmt.extend_from_slice(&1u16.to_le_bytes());
        fmt.extend_from_slice(&0u16.to_le_bytes());
        let mut head = riff_header();
        head.extend(chunk(b"fmt ", fmt.len() as u32, &fmt));
        head.extend(chunk(b"fact", 4, &FACT_SAMPLES.to_le_bytes()));
        head.extend(chunk(b"data", 1000, &[0; 1000]));

        let (bus, handle) = g2g_core::Bus::new(4);
        let mut parse = parser().with_bus(handle);
        let mut out = CollectSink::default();
        push(&mut parse, head, &mut out).await;
        assert_eq!(
            bus.try_recv(),
            Some(BusMessage::DurationChanged {
                duration_ns: 2 * NANOS_PER_SECOND
            })
        );
    }

    #[tokio::test]
    async fn an_rf64_file_reads_its_data_size_from_ds64() {
        const SAMPLES_BYTES: usize = 4 * FRAME_LEN as usize;
        let mut ds64 = Vec::new();
        ds64.extend_from_slice(&0u64.to_le_bytes()); // riffSize
        ds64.extend_from_slice(&(SAMPLES_BYTES as u64).to_le_bytes()); // dataSize
        ds64.extend_from_slice(&4u64.to_le_bytes()); // sampleCount
        ds64.extend_from_slice(&0u32.to_le_bytes()); // tableLength
        let mut file = chunk(b"RF64", u32::MAX, b"WAVE");
        file.extend(chunk(b"ds64", ds64.len() as u32, &ds64));
        file.extend(pcm_fmt(BYTE_RATE));
        file.extend(chunk(b"data", u32::MAX, &[0; SAMPLES_BYTES]));
        file.extend(chunk(b"LIST", 4, b"INFO"));
        assert_eq!(
            crate::typefind::sniff(&file),
            Some(ByteStreamEncoding::Wav),
            "RF64 types as WAV"
        );

        let mut parse = parser();
        let mut out = CollectSink::default();
        push(&mut parse, file, &mut out).await;
        assert_eq!(
            data_frames(&out),
            [(0, SAMPLES_BYTES)],
            "the LIST chunk is not samples"
        );
    }

    #[tokio::test]
    async fn info_and_id3_chunks_on_either_side_of_data_post_tags() {
        const SAMPLES_BYTES: usize = 4 * FRAME_LEN as usize;
        let mut info = b"INFO".to_vec();
        info.extend(chunk(b"INAM", 6, b"Title\0"));
        let id3 = crate::id3::write_id3v2(
            &TagList::from_iter([g2g_core::Tag::Artist("Someone".into())]),
            crate::id3::ID3V2_VERSION_2_4,
        );
        let mut file = riff_header();
        file.extend(pcm_fmt(BYTE_RATE));
        file.extend(chunk(b"LIST", info.len() as u32, &info));
        // an odd-sized data chunk, so a pad byte sits ahead of the trailing tag chunk
        file.extend(chunk(
            b"data",
            SAMPLES_BYTES as u32 + 1,
            &[0; SAMPLES_BYTES + 1],
        ));
        file.push(0);
        file.extend(chunk(b"id3 ", id3.len() as u32, &id3));
        if id3.len() % 2 == 1 {
            file.push(0);
        }

        let (bus, handle) = g2g_core::Bus::new(8);
        let mut parse = parser().with_bus(handle);
        let mut out = CollectSink::default();
        push(&mut parse, file, &mut out).await;
        parse
            .process(PipelinePacket::Eos, &mut out)
            .await
            .expect("a file whose data is followed by tags is whole");
        assert_eq!(data_frames(&out), [(0, SAMPLES_BYTES)]);
        let tags: Vec<g2g_core::Tag> = core::iter::from_fn(|| bus.try_recv())
            .filter_map(|message| match message {
                BusMessage::Tag { tags, .. } => Some(tags.tags().to_vec()),
                _ => None,
            })
            .flatten()
            .collect();
        assert_eq!(
            tags,
            [
                g2g_core::Tag::Title("Title".into()),
                g2g_core::Tag::Artist("Someone".into())
            ]
        );
    }

    #[tokio::test]
    async fn a_stream_that_ends_before_its_data_chunk_fails() {
        let mut parse = parser();
        let mut out = CollectSink::default();
        let mut head = riff_header();
        head.extend(pcm_fmt(BYTE_RATE));
        push(&mut parse, head, &mut out).await;
        assert_eq!(
            parse.process(PipelinePacket::Eos, &mut out).await,
            Err(G2gError::CapsMismatch)
        );
    }

    #[tokio::test]
    async fn a_chunk_ahead_of_data_is_skipped_without_buffering() {
        const PUSHED: usize = 64 * 1024;
        let mut parse = parser();
        let mut out = CollectSink::default();
        let mut head = riff_header();
        head.extend(pcm_fmt(BYTE_RATE));
        head.extend(chunk(b"LIST", u32::MAX - 1, &[]));
        push(&mut parse, head, &mut out).await;
        push(&mut parse, alloc::vec![0; PUSHED], &mut out).await;
        assert!(parse.buf.is_empty());
        assert_eq!(parse.chunk_bytes_to_skip, u32::MAX as usize - 1 - PUSHED);
    }

    #[tokio::test]
    async fn a_zero_length_data_chunk_runs_to_the_end() {
        const SAMPLES_BYTES: usize = 4 * FRAME_LEN as usize;
        let mut parse = parser();
        let mut out = CollectSink::default();
        let mut head = riff_header();
        head.extend(pcm_fmt(BYTE_RATE));
        head.extend(chunk(b"data", 0, &[0; SAMPLES_BYTES]));
        push(&mut parse, head, &mut out).await;
        assert_eq!(data_frames(&out), [(0, SAMPLES_BYTES)]);
    }
}
