//! AVI (`RIFF....AVI `) reading and writing: the container parse behind
//! [`avidemux`](crate::avidemux) and the file writer behind
//! [`avimux`](crate::avimux). Chunk walking comes from [`crate::riff`].
//!
//! Layout: a `LIST hdrl` holds the `avih` main header and one `LIST strl` per
//! stream (`strh` timing + `strf` format), a `LIST movi` holds the data chunks
//! (`NNdc` video, `NNwb` audio, where `NN` is the stream number in ASCII
//! digits), and an optional `idx1` at the end indexes every chunk with its
//! keyframe flag. A file past 4 GB is written as OpenDML: the first `RIFF AVI `
//! is followed by further `RIFF AVIX` lists, each with a `movi` of its own,
//! which [`parse`] walks as a continuation of the first. `idx1` only ever
//! indexes the leading list, and the OpenDML `indx` / `ix##` super-index is not
//! read, so chunks in an `AVIX` list carry no keyframe flag.
//!
//! Every count, size and offset here comes from the file, so nothing is trusted:
//! a stream count past what AVI can name, an `idx1` entry pointing outside
//! `movi`, and a truncated header all fail the parse rather than panicking or
//! allocating on the declared length. A list that runs past the end of the file
//! keeps the whole chunks before the cut, so a truncated download still plays.
//!
//! AVI stores chunks in decode order and carries no presentation timestamps, so
//! a stream's timing is reconstructed from `strh`: sample `n` starts at
//! `n * dwScale / dwRate` seconds, where `n` counts chunks when `dwSampleSize`
//! is 0 and `bytes / dwSampleSize` otherwise.

use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::caps::greatest_common_divisor;
use g2g_core::{AudioFormat, G2gError, RawVideoFormat, Tag, TagList, VideoCodec};

use crate::uncompressedvideo::{avi_raw_video, RawVideoStorage};

use crate::riff::{
    chunk_len, chunks, chunks_to_cut, padded_len, put_chunk, put_chunk_header, read_fourcc,
    read_u16, read_u32, Chunk, FourCc, CHUNK_HEADER_LEN, FOURCC_LEN, LIST_FOURCC, RIFF_FOURCC,
    RIFF_HEADER_LEN,
};
use crate::waveformat::{
    audio_format, format_tag_name, sample_frame_len, tagged_audio, wave_format_tag, WaveFormatEx,
};

/// The form type of the first `RIFF` list.
const AVI_FOURCC: FourCc = *b"AVI ";
/// The form type of every `RIFF` list past the first in an OpenDML file.
const AVIX_FOURCC: FourCc = *b"AVIX";
/// `LIST` form types this reads.
const HDRL_FOURCC: FourCc = *b"hdrl";
const STRL_FOURCC: FourCc = *b"strl";
const MOVI_FOURCC: FourCc = *b"movi";
const REC_FOURCC: FourCc = *b"rec ";
const INFO_FOURCC: FourCc = *b"INFO";
/// Chunk ids this reads.
const AVIH_FOURCC: FourCc = *b"avih";
const STRH_FOURCC: FourCc = *b"strh";
const STRF_FOURCC: FourCc = *b"strf";
const IDX1_FOURCC: FourCc = *b"idx1";

/// `strh` `fccType` values: a video and an audio stream.
const STREAM_TYPE_VIDEO: FourCc = *b"vids";
const STREAM_TYPE_AUDIO: FourCc = *b"auds";

/// Field offsets in an `avih` body.
const AVIH_STREAM_COUNT: usize = 24;
/// The fields of `avih` this reads run to the stream count.
const AVIH_MIN_LEN: usize = AVIH_STREAM_COUNT + 4;

/// Field offsets in a `strh` body.
const STRH_TYPE: usize = 0;
const STRH_HANDLER: usize = 4;
const STRH_SCALE: usize = 20;
const STRH_RATE: usize = 24;
const STRH_START: usize = 28;
const STRH_SAMPLE_SIZE: usize = 44;
/// The fields of `strh` this reads run to `dwSampleSize`.
const STRH_MIN_LEN: usize = STRH_SAMPLE_SIZE + 4;

/// Field offsets in a video `strf` (`BITMAPINFOHEADER`).
const BITMAPINFO_WIDTH: usize = 4;
const BITMAPINFO_HEIGHT: usize = 8;
const BITMAPINFO_BIT_COUNT_AT: usize = 14;
const BITMAPINFO_COMPRESSION: usize = 16;
/// A `BITMAPINFOHEADER` is 40 bytes; anything past it is codec extradata.
const BITMAPINFO_LEN: usize = 40;

/// Field offsets in an audio `strf` (`WAVEFORMATEX`).
const WAVEFORMAT_TAG: usize = 0;
const WAVEFORMAT_CHANNELS: usize = 2;
const WAVEFORMAT_SAMPLE_RATE: usize = 4;
const WAVEFORMAT_BITS: usize = 14;
const WAVEFORMAT_EXTRA_SIZE: usize = 16;
/// A `WAVEFORMATEX` without its `cbSize` field, the shortest form a PCM `strf`
/// takes; the extra bytes follow `cbSize`.
const WAVEFORMAT_LEN: usize = WAVEFORMAT_EXTRA_SIZE;
const WAVEFORMAT_EX_LEN: usize = WAVEFORMAT_EXTRA_SIZE + 2;

/// One `idx1` entry: chunk id, flags, offset, length.
const IDX1_ENTRY_LEN: usize = 16;
const IDX1_ENTRY_FLAGS: usize = 4;
const IDX1_ENTRY_OFFSET: usize = 8;
/// `AVIIF_KEYFRAME`: the indexed chunk opens a decodable point.
const AVIIF_KEYFRAME: u32 = 0x10;

/// AVI names a stream by two ASCII digits in its chunk id, so a file can hold
/// no more than this many streams whatever `avih` claims.
const MAX_STREAMS: u32 = 100;

/// `idx1` offsets and the 32-bit `RIFF` sizes cannot address past this much
/// `movi`, which is what OpenDML's `AVIX` continuation lists exist to get
/// around. [`AviWriter`] refuses to exceed it rather than write a file whose
/// index is wrong.
pub(crate) const MAX_MOVI_LEN: usize = 1 << 30;

const NANOS_PER_SECOND: u64 = 1_000_000_000;
const MICROS_PER_SECOND: u64 = 1_000_000;

/// The `BITMAPINFOHEADER` `biCompression` codes g2g decodes, matched after
/// upper-casing so `h264` and `xvid` land with their upper-case spellings.
const VIDEO_FOURCCS: &[(FourCc, VideoCodec)] = &[
    (*b"MJPG", VideoCodec::Mjpeg),
    (*b"H264", VideoCodec::H264),
    (*b"X264", VideoCodec::H264),
    (*b"AVC1", VideoCodec::H264),
    (*b"XVID", VideoCodec::Mpeg4Part2),
    (*b"DIVX", VideoCodec::Mpeg4Part2),
    (*b"DX50", VideoCodec::Mpeg4Part2),
    (*b"FMP4", VideoCodec::Mpeg4Part2),
    (*b"MP4V", VideoCodec::Mpeg4Part2),
    (*b"VP60", VideoCodec::Vp6Plain),
    (*b"VP61", VideoCodec::Vp6Plain),
    (*b"VP62", VideoCodec::Vp6Plain),
];

/// Builds the [`Tag`] one `INFO` chunk carries from its text.
type InfoTagBuilder = fn(alloc::string::String) -> Tag;

/// The `INFO` chunk ids g2g reads and the tag each carries.
const INFO_TAGS: &[(FourCc, InfoTagBuilder)] = &[
    (*b"INAM", Tag::Title),
    (*b"IART", Tag::Artist),
    (*b"IPRD", Tag::Album),
    (*b"ISFT", Tag::Encoder),
    (*b"ICMT", Tag::Comment),
];

/// The elementary format a stream's `strf` describes, or `Unsupported` for a
/// `strh` type or codec g2g has no element for. An unsupported stream keeps its
/// slot (AVI numbers streams by position) but its chunks are dropped, so the
/// streams a file does carry still play.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AviStreamKind {
    Video {
        codec: VideoCodec,
        width: u32,
        height: u32,
    },
    RawVideo {
        format: RawVideoFormat,
        width: u32,
        height: u32,
        storage: RawVideoStorage,
    },
    Audio {
        format: AudioFormat,
        channels: u8,
        sample_rate: u32,
    },
    Unsupported {
        // the fourcc or format tag of an audio or video stream g2g does not decode
        codec: Option<String>,
    },
}

/// One stream's `strh` timing and `strf` format.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AviStream {
    pub kind: AviStreamKind,
    /// `dwScale` / `dwRate`: one sample lasts `scale / rate` seconds.
    pub scale: u32,
    pub rate: u32,
    /// `dwStart`: the stream's first sample sits this many samples in.
    pub start: u32,
    /// `dwSampleSize`: bytes per sample, `0` when one chunk is one sample.
    pub sample_size: u32,
    /// The `strf` bytes past its fixed header: H.264 extradata for a video
    /// stream, the AAC `AudioSpecificConfig` for an audio one.
    pub codec_config: Vec<u8>,
}

impl AviStream {
    /// The presentation time of the sample at `sample_pos`, and how long
    /// `samples` of them last. Saturating: a `dwRate` of 0 or an absurd
    /// `dwScale` yields a clamped time rather than a divide by zero or a wrap.
    pub(crate) fn timing(&self, sample_pos: u64, samples: u64) -> (u64, u64) {
        let rate = self.rate.max(1) as u128;
        let scale = self.scale as u128;
        let to_ns = |count: u128| -> u64 {
            u64::try_from(
                count
                    .saturating_mul(scale)
                    .saturating_mul(NANOS_PER_SECOND as u128)
                    / rate,
            )
            .unwrap_or(u64::MAX)
        };
        let pts = to_ns(sample_pos as u128 + self.start as u128);
        (pts, to_ns(samples as u128))
    }

    /// How many samples a chunk of `len` bytes holds: one when the stream
    /// measures itself in chunks, else the whole samples the bytes cover.
    pub(crate) fn samples_in(&self, len: usize) -> u64 {
        match self.sample_size {
            0 => 1,
            size => len as u64 / size as u64,
        }
    }
}

/// One data chunk in `movi`: which stream it belongs to, where its payload sits
/// in the file, and whether `idx1` marked it a keyframe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AviChunk {
    pub stream: usize,
    pub body: core::ops::Range<usize>,
    pub keyframe: bool,
    // a `##pc` chunk, whose new palette entries apply from the stream's next chunk on
    pub palette_change: bool,
}

// first entry, entry count and flags, then r, g, b and flags per entry
const PALETTE_CHANGE_HEADER_LEN: usize = 4;
const PALETTE_CHANGE_ENTRY_LEN: usize = 4;
// avidec reads a longer `##pc` chunk as data
const PALETTE_CHANGE_MAX_LEN: usize = PALETTE_CHANGE_HEADER_LEN + 256 * PALETTE_CHANGE_ENTRY_LEN;

// the entries a `##pc` chunk replaces, a count of 0 meaning 256, none when the range wraps past entry 255 as in avidec
pub(crate) fn palette_change(body: &[u8]) -> Vec<(u8, [u8; 3])> {
    let (Some(&first), Some(&count)) = (body.first(), body.get(1)) else {
        return Vec::new();
    };
    let last = first.wrapping_add(count).wrapping_sub(1);
    let (entries, _) = body
        .get(PALETTE_CHANGE_HEADER_LEN..)
        .unwrap_or_default()
        .as_chunks::<PALETTE_CHANGE_ENTRY_LEN>();
    (first..=last)
        .zip(entries)
        .map(|(index, [r, g, b, _])| (index, [*r, *g, *b]))
        .collect()
}

/// A parsed AVI file: its streams in `hdrl` order, its `movi` chunks in file
/// order, and the `LIST INFO` metadata.
#[derive(Debug, Clone, Default)]
pub(crate) struct AviFile {
    pub streams: Vec<AviStream>,
    pub chunks: Vec<AviChunk>,
    pub tags: TagList,
}

/// The codec a `biCompression` names, or `None` for one g2g does not decode.
fn video_codec(tag: FourCc) -> Option<VideoCodec> {
    let upper = tag.map(|b| b.to_ascii_uppercase());
    VIDEO_FOURCCS
        .iter()
        .find(|(fourcc, _)| *fourcc == upper)
        .map(|(_, codec)| *codec)
}

/// The `biCompression` to write for a g2g video codec, or `None` for one AVI
/// cannot carry.
fn video_fourcc(codec: VideoCodec) -> Option<FourCc> {
    VIDEO_FOURCCS
        .iter()
        .find(|(_, c)| *c == codec)
        .map(|(fourcc, _)| *fourcc)
}

/// The stream number a `movi` chunk id names (`00dc` -> 0), or `None` when the
/// id does not open with two ASCII digits (`JUNK`, `LIST`, a writer's own tag).
fn chunk_stream_number(id: FourCc) -> Option<usize> {
    let tens = (id[0] as char).to_digit(10)? as usize;
    let ones = (id[1] as char).to_digit(10)? as usize;
    Some(tens * 10 + ones)
}

/// The two-digit chunk id for a stream: `NNdc` for video, `NNwb` for audio.
fn chunk_id(stream: usize, video: bool) -> FourCc {
    let suffix = if video { b"dc" } else { b"wb" };
    [
        b'0' + (stream / 10) as u8,
        b'0' + (stream % 10) as u8,
        suffix[0],
        suffix[1],
    ]
}

/// Read a `strh` + `strf` pair out of one `LIST strl`. Returns `None` when
/// either is missing or too short; the caller then keeps the stream's slot as
/// `Unsupported` so the numbering of later streams is unchanged.
fn parse_strl(data: &[u8], range: core::ops::Range<usize>) -> Option<AviStream> {
    let mut header = None;
    let mut format = None;
    for chunk in chunks(data, range) {
        match chunk.id {
            STRH_FOURCC => header = Some(chunk),
            STRF_FOURCC => format = Some(chunk),
            _ => {}
        }
    }
    let strh = header?.body(data);
    let strf = format?.body(data);
    if strh.len() < STRH_MIN_LEN {
        return None;
    }
    let stream_type = read_fourcc(strh, STRH_TYPE)?;
    let kind = match stream_type {
        STREAM_TYPE_VIDEO => parse_video_format(strh, strf),
        STREAM_TYPE_AUDIO if read_fourcc(strh, STRH_HANDLER) == Some(XAN_DPCM_HANDLER) => None,
        STREAM_TYPE_AUDIO => parse_audio_format(strf),
        _ => None,
    };
    let (kind, codec_config) = kind.unwrap_or_else(|| {
        let codec = unsupported_codec_name(stream_type, strh, strf);
        (AviStreamKind::Unsupported { codec }, Vec::new())
    });
    Some(AviStream {
        kind,
        scale: read_u32(strh, STRH_SCALE)?,
        rate: read_u32(strh, STRH_RATE)?,
        start: read_u32(strh, STRH_START)?,
        sample_size: read_u32(strh, STRH_SAMPLE_SIZE)?,
        codec_config,
    })
}

// Wing Commander IV audio: Xan DPCM behind a PCM format tag, told apart only by the handler
const XAN_DPCM_HANDLER: FourCc = *b"Axan";

fn unsupported_codec_name(stream_type: FourCc, strh: &[u8], strf: &[u8]) -> Option<String> {
    let handler = read_fourcc(strh, STRH_HANDLER);
    match stream_type {
        // a numeric biCompression such as BI_RLE8 has no printable fourcc
        STREAM_TYPE_VIDEO => [read_fourcc(strf, BITMAPINFO_COMPRESSION), handler]
            .into_iter()
            .flatten()
            .find_map(|fourcc| printable_fourcc(&fourcc))
            .or_else(|| {
                read_u32(strf, BITMAPINFO_COMPRESSION)
                    .map(|compression| alloc::format!("video compression {compression:#x}"))
            }),
        STREAM_TYPE_AUDIO if handler == Some(XAN_DPCM_HANDLER) => {
            printable_fourcc(&XAN_DPCM_HANDLER)
        }
        STREAM_TYPE_AUDIO => read_u16(strf, WAVEFORMAT_TAG).map(format_tag_name),
        _ => None,
    }
}

fn printable_fourcc(fourcc: &FourCc) -> Option<String> {
    let name: String = fourcc
        .iter()
        .filter(|byte| byte.is_ascii_graphic())
        .map(|&byte| char::from(byte))
        .collect();
    (!name.is_empty()).then_some(name)
}

/// A video stream's geometry and codec from its `BITMAPINFOHEADER`, plus the
/// extradata past it (H.264 parameter sets when the writer put them there).
/// `biCompression` falls back to the `strh` `fccHandler` some writers fill in
/// instead. The sign of `biHeight` says which way the rows run (positive is
/// bottom-up), which changes no decoded geometry, so the magnitude is what
/// travels and an uncompressed RGB stream records the row order.
fn parse_video_format(strh: &[u8], strf: &[u8]) -> Option<(AviStreamKind, Vec<u8>)> {
    if strf.len() < BITMAPINFO_LEN {
        return None;
    }
    let compression = read_fourcc(strf, BITMAPINFO_COMPRESSION)?;
    let width = read_u32(strf, BITMAPINFO_WIDTH)?;
    let signed_height = read_u32(strf, BITMAPINFO_HEIGHT)? as i32;
    let height = signed_height.unsigned_abs();
    if width == 0 || height == 0 {
        return None;
    }
    let bit_count = read_u16(strf, BITMAPINFO_BIT_COUNT_AT)?;
    let raw = avi_raw_video(compression, bit_count, signed_height > 0).or_else(|| {
        crate::uncompressedvideo::raw_video_format(compression)
            .map(|format| (format, RawVideoStorage::Tight))
    });
    if let Some((format, storage)) = raw {
        return Some((
            AviStreamKind::RawVideo {
                format,
                width,
                height,
                storage,
            },
            Vec::new(),
        ));
    }
    let typed =
        video_codec(compression).or_else(|| read_fourcc(strh, STRH_HANDLER).and_then(video_codec));
    let (codec, codec_config) = match typed {
        Some(codec) => (codec, Vec::from(&strf[BITMAPINFO_LEN..])),
        None => (tagged_video(compression)?, strf.to_vec()),
    };
    Some((
        AviStreamKind::Video {
            codec,
            width,
            height,
        },
        codec_config,
    ))
}

// a codec no typed mapping names goes on under its tag, its whole `strf` as the config
#[cfg(feature = "ffmpeg-long-tail")]
fn tagged_video(compression: FourCc) -> Option<VideoCodec> {
    Some(VideoCodec::Tagged(g2g_core::CodecTag {
        namespace: g2g_core::TagNamespace::Riff,
        tag: u32::from_le_bytes(compression),
    }))
}

#[cfg(not(feature = "ffmpeg-long-tail"))]
fn tagged_video(_compression: FourCc) -> Option<VideoCodec> {
    None
}

/// An audio stream's format from its `WAVEFORMATEX`, plus the `cbSize` extra
/// bytes (an AAC stream's `AudioSpecificConfig`), bounded by what the chunk
/// actually holds rather than by the declared `cbSize`.
fn parse_audio_format(strf: &[u8]) -> Option<(AviStreamKind, Vec<u8>)> {
    if strf.len() < WAVEFORMAT_LEN {
        return None;
    }
    let tag = read_u16(strf, WAVEFORMAT_TAG)?;
    let typed = audio_format(tag, read_u16(strf, WAVEFORMAT_BITS)?);
    let channels = read_u16(strf, WAVEFORMAT_CHANNELS)?;
    let sample_rate = read_u32(strf, WAVEFORMAT_SAMPLE_RATE)?;
    if channels == 0 || channels > u8::MAX as u16 || sample_rate == 0 {
        return None;
    }
    let Some(format) = typed else {
        let kind = AviStreamKind::Audio {
            format: tagged_audio(tag)?,
            channels: channels as u8,
            sample_rate,
        };
        return Some((kind, strf.to_vec()));
    };
    let extra = match read_u16(strf, WAVEFORMAT_EXTRA_SIZE) {
        Some(declared) => {
            let end = WAVEFORMAT_EX_LEN
                .saturating_add(declared as usize)
                .min(strf.len());
            Vec::from(strf.get(WAVEFORMAT_EX_LEN..end).unwrap_or_default())
        }
        None => Vec::new(),
    };
    Some((
        AviStreamKind::Audio {
            format,
            channels: channels as u8,
            sample_rate,
        },
        extra,
    ))
}

/// The `LIST INFO` metadata as tags. Each body is a null-padded string.
fn parse_info(data: &[u8], range: core::ops::Range<usize>) -> TagList {
    let mut tags = TagList::new();
    for chunk in chunks(data, range) {
        let Some((_, build)) = INFO_TAGS.iter().find(|(id, _)| *id == chunk.id) else {
            continue;
        };
        let body = chunk.body(data);
        let text = core::str::from_utf8(body)
            .unwrap_or_default()
            .trim_end_matches('\0');
        if !text.is_empty() {
            tags.push(build(text.into()));
        }
    }
    tags
}

/// Collect the data chunks of one `LIST movi`, descending into the `LIST rec `
/// groups an interleaved file wraps them in. Empty chunks are dropped: a writer
/// emits them as padding and no decoder has anything to do with them. A data
/// chunk cut short by the end of the file is dropped too, unless it belongs to a
/// long-tail video stream.
fn collect_movi(
    data: &[u8],
    range: core::ops::Range<usize>,
    streams: &[AviStream],
    out: &mut Vec<AviChunk>,
) {
    let mut walk = chunks_to_cut(data, range);
    while let Some(chunk) = walk.next() {
        if chunk.list_form(data) == Some(REC_FOURCC) {
            collect_data_chunks(data, chunk.list_body(), streams, out);
            continue;
        }
        if walk.overran() {
            push_cut_chunk(chunk, streams, out);
            break;
        }
        push_data_chunk(chunk, streams.len(), out);
    }
}

// AVI only puts `rec ` directly in `movi`, so a list inside one is skipped rather than walked
fn collect_data_chunks(
    data: &[u8],
    range: core::ops::Range<usize>,
    streams: &[AviStream],
    out: &mut Vec<AviChunk>,
) {
    let mut walk = chunks_to_cut(data, range);
    while let Some(chunk) = walk.next() {
        if walk.overran() {
            push_cut_chunk(chunk, streams, out);
            break;
        }
        push_data_chunk(chunk, streams.len(), out);
    }
}

// libavcodec also decodes what remains of a cut long-tail video chunk, as ffmpeg does
fn push_cut_chunk(chunk: Chunk, streams: &[AviStream], out: &mut Vec<AviChunk>) {
    let tagged_video = chunk_stream_number(chunk.id)
        .and_then(|stream| streams.get(stream))
        .is_some_and(|stream| {
            matches!(
                stream.kind,
                AviStreamKind::Video {
                    codec: VideoCodec::Tagged(_),
                    ..
                }
            )
        });
    if tagged_video {
        push_data_chunk(chunk, streams.len(), out);
    }
}

fn push_data_chunk(chunk: Chunk, stream_count: usize, out: &mut Vec<AviChunk>) {
    let Some(stream) = chunk_stream_number(chunk.id) else {
        return;
    };
    if stream >= stream_count || chunk.body.is_empty() {
        return;
    }
    let palette_change = chunk.id[2..] == *b"pc" && chunk.body.len() <= PALETTE_CHANGE_MAX_LEN;
    out.push(AviChunk {
        stream,
        body: chunk.body,
        keyframe: false,
        palette_change,
    });
}

/// Apply the `idx1` keyframe flags to the collected chunks. `idx1` offsets are
/// normally relative to the position of the `movi` form type, but some writers
/// store absolute file offsets; the base is chosen by which one lands on a
/// chunk this walk already found. An entry that matches nothing is ignored, so
/// a stale or truncated index costs keyframe flags rather than the parse.
fn apply_idx1(data: &[u8], idx1: &[u8], movi_form_offset: usize, chunks_out: &mut [AviChunk]) {
    let entries = idx1.len() / IDX1_ENTRY_LEN;
    let body_of = |header: usize| header.checked_add(CHUNK_HEADER_LEN);
    let base = |entry: usize| -> Option<usize> {
        let offset = read_u32(idx1, entry * IDX1_ENTRY_LEN + IDX1_ENTRY_OFFSET)? as usize;
        let relative = body_of(movi_form_offset.checked_add(offset)?)?;
        if chunk_starting_at(chunks_out, relative).is_some() {
            return Some(movi_form_offset);
        }
        let absolute = body_of(offset)?;
        chunk_starting_at(chunks_out, absolute).map(|_| 0)
    };
    let Some(base) = (0..entries).find_map(base) else {
        return;
    };
    for entry in 0..entries {
        let at = entry * IDX1_ENTRY_LEN;
        let (Some(flags), Some(offset)) = (
            read_u32(idx1, at + IDX1_ENTRY_FLAGS),
            read_u32(idx1, at + IDX1_ENTRY_OFFSET),
        ) else {
            continue;
        };
        if flags & AVIIF_KEYFRAME == 0 {
            continue;
        }
        let Some(body) = base
            .checked_add(offset as usize)
            .and_then(body_of)
            .filter(|start| *start < data.len())
        else {
            continue;
        };
        if let Some(index) = chunk_starting_at(chunks_out, body) {
            chunks_out[index].keyframe = true;
        }
    }
}

/// The chunk whose body starts at `start`. `collect_movi` emits chunks in file
/// order, so the bodies are sorted by offset.
fn chunk_starting_at(chunks_out: &[AviChunk], start: usize) -> Option<usize> {
    chunks_out
        .binary_search_by_key(&start, |chunk| chunk.body.start)
        .ok()
}

/// Parse a whole AVI file. Fails with [`G2gError::CapsMismatch`] on anything
/// malformed: a missing or truncated `RIFF AVI ` header, an `hdrl` the file does
/// not hold whole, a stream count AVI cannot name, or a file with no stream g2g
/// can carry. A `RIFF`, `movi` or `idx1` cut short by the end of the file is
/// read up to the cut.
pub(crate) fn parse(data: &[u8]) -> Result<AviFile, G2gError> {
    if read_fourcc(data, 0) != Some(RIFF_FOURCC)
        || read_fourcc(data, CHUNK_HEADER_LEN) != Some(AVI_FOURCC)
    {
        return Err(G2gError::CapsMismatch);
    }
    let mut file = AviFile::default();
    let mut movi_ranges: Vec<(usize, core::ops::Range<usize>)> = Vec::new();
    let mut idx1: Option<core::ops::Range<usize>> = None;

    // Every `RIFF` list in the file: the leading `AVI ` one, then the `AVIX`
    // continuations OpenDML appends past the 4 GB the first list can address.
    let mut riff_at = 0usize;
    while data.len().saturating_sub(riff_at) >= RIFF_HEADER_LEN {
        // anything that is not a `RIFF` list after the last one is trailing padding
        let Some(riff) = chunks_to_cut(data, riff_at..data.len())
            .next()
            .filter(|c| c.id == RIFF_FOURCC)
        else {
            break;
        };
        if riff_at != 0 && read_fourcc(data, riff.body.start) != Some(AVIX_FOURCC) {
            break;
        }
        for chunk in chunks_to_cut(data, riff.list_body()) {
            match chunk.list_form(data) {
                Some(HDRL_FOURCC) if file.streams.is_empty() => {
                    parse_hdrl(data, chunk.list_body(), &mut file)?;
                }
                Some(MOVI_FOURCC) => {
                    // The `movi` form type sits one fourcc into the list body,
                    // which is what `idx1` measures its offsets from.
                    movi_ranges.push((chunk.body.start, chunk.list_body()));
                }
                Some(INFO_FOURCC) if file.tags.is_empty() => {
                    file.tags = parse_info(data, chunk.list_body());
                }
                _ if chunk.id == IDX1_FOURCC && idx1.is_none() => idx1 = Some(chunk.body),
                _ => {}
            }
        }
        riff_at = riff
            .body
            .end
            .checked_add((riff.body.end - riff.body.start) % 2)
            .ok_or(G2gError::CapsMismatch)?;
    }

    if file.streams.is_empty() {
        return Err(G2gError::CapsMismatch);
    }
    for (_, range) in &movi_ranges {
        collect_movi(data, range.clone(), &file.streams, &mut file.chunks);
    }
    if let (Some(idx1), Some((form_offset, _))) = (idx1, movi_ranges.first()) {
        apply_idx1(
            data,
            data.get(idx1).unwrap_or_default(),
            *form_offset,
            &mut file.chunks,
        );
    } else {
        // With no index, only an all-intra codec can promise every chunk is a
        // decodable point; an inter-coded stream keeps its flags clear.
        for chunk in file.chunks.iter_mut() {
            chunk.keyframe = matches!(
                file.streams[chunk.stream].kind,
                AviStreamKind::Video {
                    codec: VideoCodec::Mjpeg,
                    ..
                } | AviStreamKind::RawVideo { .. }
            );
        }
    }
    Ok(file)
}

// the streams alone, from a file prefix that may end anywhere past `hdrl`
pub(crate) fn parse_streams(data: &[u8]) -> Result<Vec<AviStream>, G2gError> {
    if read_fourcc(data, 0) != Some(RIFF_FOURCC)
        || read_fourcc(data, CHUNK_HEADER_LEN) != Some(AVI_FOURCC)
    {
        return Err(G2gError::CapsMismatch);
    }
    let hdrl = chunks(data, RIFF_HEADER_LEN..data.len())
        .find(|chunk| chunk.list_form(data) == Some(HDRL_FOURCC))
        .ok_or(G2gError::CapsMismatch)?;
    let mut file = AviFile::default();
    parse_hdrl(data, hdrl.list_body(), &mut file)?;
    if file.streams.is_empty() {
        return Err(G2gError::CapsMismatch);
    }
    Ok(file.streams)
}

/// Read the `avih` header and one stream per `LIST strl`.
fn parse_hdrl(
    data: &[u8],
    range: core::ops::Range<usize>,
    file: &mut AviFile,
) -> Result<(), G2gError> {
    let mut walk = chunks(data, range);
    for chunk in walk.by_ref() {
        if chunk.id == AVIH_FOURCC {
            let avih = chunk.body(data);
            if avih.len() < AVIH_MIN_LEN {
                return Err(G2gError::CapsMismatch);
            }
            // The declared count is checked before anything is sized from the
            // header, so a file claiming a million streams fails here rather
            // than after the walk has allocated for them.
            let declared = read_u32(avih, AVIH_STREAM_COUNT).ok_or(G2gError::CapsMismatch)?;
            if declared > MAX_STREAMS {
                return Err(G2gError::CapsMismatch);
            }
        }
        if chunk.list_form(data) == Some(STRL_FOURCC) {
            if file.streams.len() as u32 >= MAX_STREAMS {
                return Err(G2gError::CapsMismatch);
            }
            file.streams
                .push(parse_strl(data, chunk.list_body()).unwrap_or(AviStream {
                    kind: AviStreamKind::Unsupported { codec: None },
                    scale: 1,
                    rate: 1,
                    start: 0,
                    sample_size: 0,
                    codec_config: Vec::new(),
                }));
        }
    }
    if walk.overran() {
        return Err(G2gError::CapsMismatch);
    }
    Ok(())
}

/// One stream an [`AviWriter`] writes.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AviWriteStream {
    Video {
        codec: VideoCodec,
        width: u32,
        height: u32,
    },
    Audio {
        format: AudioFormat,
        channels: u8,
        sample_rate: u32,
    },
}

impl AviWriteStream {
    /// Whether AVI can name this stream: a `BITMAPINFOHEADER` FourCC for video,
    /// a `WAVEFORMATEX` tag for audio.
    pub(crate) fn is_writable(&self) -> bool {
        match self {
            Self::Video { codec, .. } => video_fourcc(*codec).is_some(),
            // the writer has no ADPCM block size to put in `nBlockAlign`
            Self::Audio { format, .. } => {
                *format != AudioFormat::ImaAdpcm && wave_format_tag(*format).is_some()
            }
        }
    }
}

/// A chunk queued for writing.
#[derive(Debug)]
struct PendingChunk {
    stream: usize,
    data: Vec<u8>,
    keyframe: bool,
}

/// Builds a whole AVI in memory: the header needs each stream's sample count
/// and the `idx1` needs every chunk's offset, so nothing can be written until
/// the last chunk is in hand.
///
/// Refuses to exceed [`MAX_MOVI_LEN`] of chunk data, because `idx1` and the
/// `RIFF` size are 32-bit: a larger file needs OpenDML `AVIX` continuation
/// lists, which this does not write.
#[derive(Debug)]
pub(crate) struct AviWriter {
    streams: Vec<AviWriteStream>,
    chunks: Vec<PendingChunk>,
    movi_len: usize,
    /// Per stream, the first and last pts seen and how many chunks: the sample
    /// rate a stream declares in `strh` is reconstructed from these, since AVI
    /// has no per-chunk timestamp to carry them.
    spans: Vec<Option<(u64, u64, u64)>>,
}

/// The frame period a stream declares when it carried too few chunks to measure
/// one: 25 fps, the rate the AVI main header's default `dwMicroSecPerFrame` of
/// 40000 names.
const DEFAULT_FRAME_PERIOD_NS: u64 = NANOS_PER_SECOND / 25;

impl AviWriter {
    pub(crate) fn new(streams: Vec<AviWriteStream>) -> Self {
        let spans = alloc::vec![None; streams.len()];
        Self {
            streams,
            chunks: Vec::new(),
            movi_len: 0,
            spans,
        }
    }

    /// Queue one chunk. Fails once the queued data would push `movi` past what
    /// a 32-bit index can address.
    pub(crate) fn push(
        &mut self,
        stream: usize,
        data: Vec<u8>,
        pts_ns: u64,
        keyframe: bool,
    ) -> Result<(), G2gError> {
        if stream >= self.streams.len() {
            return Err(G2gError::CapsMismatch);
        }
        let added = CHUNK_HEADER_LEN
            .checked_add(padded_len(data.len()).ok_or(G2gError::CapsMismatch)?)
            .ok_or(G2gError::CapsMismatch)?;
        self.movi_len = self
            .movi_len
            .checked_add(added)
            .ok_or(G2gError::CapsMismatch)?;
        if self.movi_len > MAX_MOVI_LEN {
            return Err(G2gError::CapsMismatch);
        }
        self.spans[stream] = Some(match self.spans[stream] {
            Some((first, _, count)) => (first, pts_ns, count + 1),
            None => (pts_ns, pts_ns, 1),
        });
        self.chunks.push(PendingChunk {
            stream,
            data,
            keyframe,
        });
        Ok(())
    }

    /// The mean interval between a stream's chunks, or the default when it
    /// carried fewer than two.
    fn chunk_period_ns(&self, stream: usize) -> u64 {
        match self.spans[stream] {
            Some((first, last, count)) if count > 1 && last > first => (last - first) / (count - 1),
            _ => DEFAULT_FRAME_PERIOD_NS,
        }
    }

    /// The number of chunks written for a stream.
    fn chunk_count(&self, stream: usize) -> u32 {
        self.spans[stream]
            .map(|(_, _, count)| count.min(u32::MAX as u64) as u32)
            .unwrap_or(0)
    }

    /// Serialize the file: `RIFF AVI ` + `LIST hdrl` + `LIST movi` + `idx1`.
    pub(crate) fn finish(&self) -> Result<Vec<u8>, G2gError> {
        let mut hdrl = Vec::from(HDRL_FOURCC);
        put_chunk(&mut hdrl, &AVIH_FOURCC, &self.avih());
        for stream in 0..self.streams.len() {
            let mut strl = Vec::from(STRL_FOURCC);
            put_chunk(&mut strl, &STRH_FOURCC, &self.strh(stream)?);
            put_chunk(&mut strl, &STRF_FOURCC, &self.strf(stream)?);
            put_chunk(&mut hdrl, &LIST_FOURCC, &strl);
        }

        // `push` caps `movi_len`, so none of these sums can overflow.
        let movi_body_len = FOURCC_LEN + self.movi_len;
        let idx1_len = self.chunks.len() * IDX1_ENTRY_LEN;
        let riff_body_len =
            FOURCC_LEN + chunk_len(hdrl.len()) + chunk_len(movi_body_len) + chunk_len(idx1_len);
        let mut file = Vec::with_capacity(CHUNK_HEADER_LEN + riff_body_len);
        put_chunk_header(&mut file, &RIFF_FOURCC, riff_body_len);
        file.extend_from_slice(&AVI_FOURCC);
        put_chunk(&mut file, &LIST_FOURCC, &hdrl);
        put_chunk_header(&mut file, &LIST_FOURCC, movi_body_len);
        // `idx1` measures from the `movi` form type.
        let movi_form_at = file.len();
        file.extend_from_slice(&MOVI_FOURCC);
        let mut idx1 = Vec::with_capacity(idx1_len);
        for chunk in &self.chunks {
            let video = matches!(self.streams[chunk.stream], AviWriteStream::Video { .. });
            let id = chunk_id(chunk.stream, video);
            idx1.extend_from_slice(&id);
            idx1.extend_from_slice(
                &(if chunk.keyframe { AVIIF_KEYFRAME } else { 0 }).to_le_bytes(),
            );
            idx1.extend_from_slice(&((file.len() - movi_form_at) as u32).to_le_bytes());
            idx1.extend_from_slice(&(chunk.data.len() as u32).to_le_bytes());
            put_chunk(&mut file, &id, &chunk.data);
        }
        put_chunk(&mut file, &IDX1_FOURCC, &idx1);
        debug_assert_eq!(file.len(), CHUNK_HEADER_LEN + riff_body_len);
        Ok(file)
    }

    /// The main header. `dwMicroSecPerFrame` and `dwTotalFrames` describe the
    /// video stream; an audio-only file leaves them at the defaults.
    fn avih(&self) -> Vec<u8> {
        let video = self
            .streams
            .iter()
            .position(|s| matches!(s, AviWriteStream::Video { .. }));
        let period_ns = video.map_or(DEFAULT_FRAME_PERIOD_NS, |s| self.chunk_period_ns(s));
        let (width, height) = match video.map(|s| &self.streams[s]) {
            Some(AviWriteStream::Video { width, height, .. }) => (*width, *height),
            _ => (0, 0),
        };
        let mut out = Vec::new();
        let mut put = |v: u32| out.extend_from_slice(&v.to_le_bytes());
        put((period_ns / (NANOS_PER_SECOND / MICROS_PER_SECOND)) as u32);
        put(0); // dwMaxBytesPerSec, advisory
        put(0); // dwPaddingGranularity
        put(AVIF_HASINDEX);
        put(video.map_or(0, |s| self.chunk_count(s)));
        put(0); // dwInitialFrames
        put(self.streams.len() as u32);
        put(0); // dwSuggestedBufferSize, advisory
        put(width);
        put(height);
        for _ in 0..4 {
            put(0); // dwReserved
        }
        out
    }

    /// One stream's `strh`.
    fn strh(&self, stream: usize) -> Result<Vec<u8>, G2gError> {
        let period_ns = self.chunk_period_ns(stream);
        let (kind, handler, scale, rate, sample_size) = match &self.streams[stream] {
            AviWriteStream::Video { codec, .. } => {
                let fourcc = video_fourcc(*codec).ok_or(G2gError::CapsMismatch)?;
                let (scale, rate) = frame_rate_fraction(period_ns);
                (STREAM_TYPE_VIDEO, fourcc, scale, rate, 0)
            }
            AviWriteStream::Audio {
                format,
                channels,
                sample_rate,
            } => {
                // A PCM stream measures itself in sample frames, so its chunks
                // can hold any whole number of them; a compressed stream stores
                // one frame per chunk and scales by that frame's sample count.
                let (scale, size) = match sample_frame_len(*format, *channels) {
                    Some(align) => (1, align as u32),
                    None => (samples_per_frame(period_ns, *sample_rate), 0),
                };
                (STREAM_TYPE_AUDIO, [0; 4], scale, *sample_rate, size)
            }
        };
        let mut out = Vec::from(kind);
        out.extend_from_slice(&handler);
        let mut put = |v: u32| out.extend_from_slice(&v.to_le_bytes());
        put(0); // dwFlags
        put(0); // wPriority + wLanguage
        put(0); // dwInitialFrames
        put(scale);
        put(rate);
        put(0); // dwStart
        put(self.sample_length(stream, sample_size));
        put(0); // dwSuggestedBufferSize, advisory
        put(u32::MAX); // dwQuality: the codec default
        put(sample_size);
        put(0); // rcFrame left + top
        put(0); // rcFrame right + bottom
        Ok(out)
    }

    /// `dwLength`: the stream's length in the samples `dwSampleSize` names, so
    /// chunks for a compressed stream and sample frames for PCM.
    fn sample_length(&self, stream: usize, sample_size: u32) -> u32 {
        if sample_size == 0 {
            return self.chunk_count(stream);
        }
        let bytes: usize = self
            .chunks
            .iter()
            .filter(|c| c.stream == stream)
            .map(|c| c.data.len())
            .sum();
        (bytes / sample_size as usize).min(u32::MAX as usize) as u32
    }

    /// One stream's `strf`: a `BITMAPINFOHEADER` for video, a `WAVEFORMATEX`
    /// for audio.
    fn strf(&self, stream: usize) -> Result<Vec<u8>, G2gError> {
        let mut out = Vec::new();
        match &self.streams[stream] {
            AviWriteStream::Video {
                codec,
                width,
                height,
            } => {
                let fourcc = video_fourcc(*codec).ok_or(G2gError::CapsMismatch)?;
                out.extend_from_slice(&(BITMAPINFO_LEN as u32).to_le_bytes());
                out.extend_from_slice(&width.to_le_bytes());
                out.extend_from_slice(&height.to_le_bytes());
                out.extend_from_slice(&BITMAPINFO_PLANES.to_le_bytes());
                out.extend_from_slice(&BITMAPINFO_BIT_COUNT.to_le_bytes());
                out.extend_from_slice(&fourcc);
                out.extend_from_slice(&(width * height * 3).to_le_bytes()); // biSizeImage
                for _ in 0..4 {
                    out.extend_from_slice(&0u32.to_le_bytes()); // resolution + palette
                }
            }
            AviWriteStream::Audio {
                format,
                channels,
                sample_rate,
            } => {
                let (tag, bits) = wave_format_tag(*format).ok_or(G2gError::CapsMismatch)?;
                // A demuxer negotiates compressed audio on the `0/0` unknown
                // layout and refines it at runtime; without that refinement the
                // `strf` would claim a stream no reader can play.
                if *channels == 0 || *sample_rate == 0 {
                    return Err(G2gError::CapsMismatch);
                }
                let pcm_align = sample_frame_len(*format, *channels);
                let align = pcm_align.unwrap_or(1);
                let byte_rate = match pcm_align {
                    Some(align) => *sample_rate as usize * align,
                    // A compressed stream's average rate is what it actually
                    // wrote over the time its chunks covered.
                    None => self.compressed_byte_rate(stream),
                };
                let wave_format = WaveFormatEx {
                    tag,
                    channels: u16::from(*channels),
                    sample_rate: *sample_rate,
                    byte_rate: byte_rate.min(u32::MAX as usize) as u32,
                    block_align: align.min(u16::MAX as usize) as u16,
                    bits,
                };
                // AVI always writes the `cbSize` field, as ffmpeg does.
                out = wave_format.to_bytes(Some(&[]));
            }
        }
        Ok(out)
    }

    /// A compressed audio stream's `nAvgBytesPerSec`, from what it wrote over
    /// the span its chunks cover.
    fn compressed_byte_rate(&self, stream: usize) -> usize {
        let bytes: usize = self
            .chunks
            .iter()
            .filter(|c| c.stream == stream)
            .map(|c| c.data.len())
            .sum();
        let span_ns = self.chunk_period_ns(stream) * self.chunk_count(stream).max(1) as u64;
        if span_ns == 0 {
            return 0;
        }
        (bytes as u64 * NANOS_PER_SECOND / span_ns) as usize
    }
}

/// `AVIF_HASINDEX`: the file ends with an `idx1`, which this always writes.
const AVIF_HASINDEX: u32 = 0x10;
/// `biPlanes` is 1 in every `BITMAPINFOHEADER`.
const BITMAPINFO_PLANES: u16 = 1;
/// `biBitCount` for the packed 24-bit image a compressed stream nominally
/// decodes to; advisory, no decoder reads it.
const BITMAPINFO_BIT_COUNT: u16 = 24;

/// `(dwScale, dwRate)` for a frame period, reduced so the common rates print as
/// the fractions they are (40 ms -> 1/25).
fn frame_rate_fraction(period_ns: u64) -> (u32, u32) {
    let period = period_ns.clamp(1, NANOS_PER_SECOND);
    let divisor = greatest_common_divisor(period, NANOS_PER_SECOND);
    (
        (period / divisor) as u32,
        (NANOS_PER_SECOND / divisor) as u32,
    )
}

/// How many samples a compressed audio frame of `period_ns` covers at
/// `sample_rate`, at least one.
fn samples_per_frame(period_ns: u64, sample_rate: u32) -> u32 {
    let samples = (period_ns as u128 * sample_rate as u128 + (NANOS_PER_SECOND as u128 / 2))
        / NANOS_PER_SECOND as u128;
    u32::try_from(samples).unwrap_or(u32::MAX).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// A one-video-stream, one-audio-stream AVI written by [`AviWriter`], the
    /// fixture the parser round-trips against.
    fn write_av() -> Vec<u8> {
        let mut writer = AviWriter::new(vec![
            AviWriteStream::Video {
                codec: VideoCodec::Mjpeg,
                width: 32,
                height: 16,
            },
            AviWriteStream::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                sample_rate: 48_000,
            },
        ]);
        for i in 0..4u64 {
            writer
                .push(0, vec![i as u8; 5], i * DEFAULT_FRAME_PERIOD_NS, true)
                .expect("video chunk");
            writer
                .push(
                    1,
                    vec![0x7f; 4 * 48_000 / 25],
                    i * DEFAULT_FRAME_PERIOD_NS,
                    true,
                )
                .expect("audio chunk");
        }
        writer.finish().expect("the file serializes")
    }

    #[test]
    fn round_trips_a_written_file() {
        let bytes = write_av();
        let file = parse(&bytes).expect("the written file parses");
        assert_eq!(file.streams.len(), 2);
        assert_eq!(
            file.streams[0].kind,
            AviStreamKind::Video {
                codec: VideoCodec::Mjpeg,
                width: 32,
                height: 16
            }
        );
        assert_eq!(
            file.streams[1].kind,
            AviStreamKind::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                sample_rate: 48_000
            }
        );
        // 25 fps: dwScale/dwRate reduce to 1/25, so frame n lands on n * 40 ms.
        assert_eq!(file.streams[0].scale, 1);
        assert_eq!(file.streams[0].rate, 25);
        let video: Vec<&AviChunk> = file.chunks.iter().filter(|c| c.stream == 0).collect();
        assert_eq!(video.len(), 4);
        assert!(video.iter().all(|c| c.keyframe), "idx1 carries the flags");
        for (i, chunk) in video.iter().enumerate() {
            let (pts, duration) = file.streams[0].timing(i as u64, 1);
            assert_eq!(pts, i as u64 * DEFAULT_FRAME_PERIOD_NS);
            assert_eq!(duration, DEFAULT_FRAME_PERIOD_NS);
            assert_eq!(bytes[chunk.body.clone()], [i as u8; 5]);
        }
        // PCM measures itself in sample frames, so its chunk timing comes from
        // the running byte count over the block align.
        let audio = file.streams[1].clone();
        assert_eq!(audio.sample_size, 4);
        assert_eq!(
            audio.timing(audio.samples_in(4 * 48_000 / 25), 0).0,
            40_000_000
        );
    }

    #[test]
    fn reads_the_vp6_fourccs_as_plain_vp6() {
        let mut bytes = write_av();
        while let Some(at) = bytes.windows(FOURCC_LEN).position(|w| w == b"MJPG") {
            bytes[at..at + FOURCC_LEN].copy_from_slice(b"VP62");
        }
        let file = parse(&bytes).expect("the file parses");
        let AviStreamKind::Video { codec, .. } = file.streams[0].kind else {
            panic!("stream 0 is video");
        };
        assert_eq!(codec, VideoCodec::Vp6Plain);
    }

    // the written file with its video recoded to Cinepak and its audio to TrueSpeech
    fn write_unmapped_codecs() -> Vec<u8> {
        const TRUESPEECH: u16 = 0x0022;
        let mut bytes = write_av();
        while let Some(at) = bytes.windows(FOURCC_LEN).position(|w| w == b"MJPG") {
            bytes[at..at + FOURCC_LEN].copy_from_slice(b"cvid");
        }
        let audio_strf = bytes
            .windows(FOURCC_LEN)
            .enumerate()
            .filter(|(_, window)| *window == STRF_FOURCC)
            .nth(1)
            .map(|(at, _)| at + CHUNK_HEADER_LEN)
            .expect("the audio stream has a strf");
        bytes[audio_strf + WAVEFORMAT_TAG..audio_strf + WAVEFORMAT_TAG + 2]
            .copy_from_slice(&TRUESPEECH.to_le_bytes());
        bytes
    }

    #[cfg(feature = "ffmpeg-long-tail")]
    #[test]
    fn forwards_an_unmapped_codec_under_its_tag() {
        use g2g_core::{CodecTag, TagNamespace};
        let file = parse(&write_unmapped_codecs()).expect("the file parses");
        let AviStreamKind::Video { codec, .. } = file.streams[0].kind else {
            panic!("stream 0 is video");
        };
        let riff = |tag| CodecTag {
            namespace: TagNamespace::Riff,
            tag,
        };
        assert_eq!(
            codec,
            VideoCodec::Tagged(riff(u32::from_le_bytes(*b"cvid")))
        );
        let compression = BITMAPINFO_COMPRESSION..BITMAPINFO_COMPRESSION + FOURCC_LEN;
        assert_eq!(&file.streams[0].codec_config[compression], b"cvid");
        let AviStreamKind::Audio { format, .. } = file.streams[1].kind else {
            panic!("stream 1 is audio");
        };
        assert_eq!(format, AudioFormat::Tagged(riff(0x0022)));
    }

    #[cfg(feature = "ffmpeg-long-tail")]
    #[test]
    fn a_cut_long_tail_video_chunk_is_kept() {
        let bytes = write_unmapped_codecs();
        let whole = parse(&bytes).expect("the whole file parses");
        let last_video = whole
            .chunks
            .iter()
            .rev()
            .find(|chunk| chunk.stream == 0)
            .expect("a video chunk");
        let cut = last_video.body.start + last_video.body.len() / 2;
        let file = parse(&bytes[..cut]).expect("the cut file parses");
        let kept = file.chunks.last().expect("the cut file keeps chunks");
        assert_eq!(
            (kept.stream, kept.body.clone()),
            (0, last_video.body.start..cut)
        );
    }

    #[cfg(not(feature = "ffmpeg-long-tail"))]
    #[test]
    fn names_an_unmapped_codec_without_the_long_tail() {
        let file = parse(&write_unmapped_codecs()).expect("the file parses");
        let names: Vec<_> = file
            .streams
            .iter()
            .map(|stream| match &stream.kind {
                AviStreamKind::Unsupported { codec } => codec.clone(),
                _ => None,
            })
            .collect();
        assert_eq!(
            names,
            [
                Some(String::from("cvid")),
                Some(String::from("audio format tag 0x0022"))
            ]
        );
    }

    #[test]
    fn rejects_a_file_cut_inside_its_header() {
        let bytes = write_av();
        let avih = find_chunk_body(&bytes, &AVIH_FOURCC).expect("the avih");
        let hdrl_form = RIFF_HEADER_LEN + CHUNK_HEADER_LEN;
        for cut in [
            0,
            FOURCC_LEN,
            RIFF_HEADER_LEN - 1,
            hdrl_form,
            avih,
            avih + AVIH_MIN_LEN,
        ] {
            assert!(
                parse(&bytes[..cut]).is_err(),
                "a file cut to {cut} bytes must fail the parse"
            );
        }
    }

    #[test]
    fn nested_rec_lists_do_not_exhaust_the_stack() {
        const DEPTH: usize = 1 << 16;
        const LIST_HEADER_LEN: usize = CHUNK_HEADER_LEN + FOURCC_LEN;
        let video = *b"00dc";
        let payload = [1u8, 2, 3, 4];
        let mut data = Vec::new();
        for layer in 0..DEPTH {
            let inner_len =
                (DEPTH - 1 - layer) * LIST_HEADER_LEN + CHUNK_HEADER_LEN + payload.len();
            put_chunk_header(&mut data, &LIST_FOURCC, FOURCC_LEN + inner_len);
            data.extend_from_slice(&REC_FOURCC);
        }
        put_chunk(&mut data, &video, &payload);
        let top_level = data.len() + CHUNK_HEADER_LEN;
        put_chunk(&mut data, &video, &payload);

        let streams = parse(&write_av()).expect("the file parses").streams;
        let mut out = Vec::new();
        collect_movi(&data, 0..data.len(), &streams[..1], &mut out);
        let bodies: Vec<_> = out.iter().map(|chunk| chunk.body.clone()).collect();
        assert_eq!(bodies, vec![top_level..top_level + payload.len()]);
    }

    #[test]
    fn a_file_cut_inside_movi_keeps_its_whole_chunks() {
        let bytes = write_av();
        let whole = parse(&bytes).expect("the whole file parses");
        let positions = |file: &AviFile| -> Vec<(usize, core::ops::Range<usize>)> {
            file.chunks
                .iter()
                .map(|chunk| (chunk.stream, chunk.body.clone()))
                .collect()
        };
        for chunk in &whole.chunks {
            let cut = chunk.body.start + chunk.body.len() / 2;
            let file = parse(&bytes[..cut]).unwrap_or_else(|e| panic!("cut at {cut}: {e:?}"));
            assert_eq!(file.streams, whole.streams);
            let mut expected = whole.clone();
            expected.chunks.retain(|c| c.body.end <= cut);
            assert_eq!(positions(&file), positions(&expected), "cut at {cut}");
            // with no idx1 left, only the all-intra MJPEG stream claims keyframes
            for kept in &file.chunks {
                assert_eq!(kept.keyframe, kept.stream == 0, "cut at {cut}");
            }
        }
    }

    #[test]
    fn a_palette_change_reads_its_entries_like_avidec() {
        let change = |first: u8, count: u8, entries: &[[u8; 4]]| {
            let mut body = vec![first, count, 0, 0];
            for entry in entries {
                body.extend_from_slice(entry);
            }
            palette_change(&body)
        };
        assert_eq!(
            change(5, 2, &[[1, 2, 3, 0], [4, 5, 6, 0]]),
            vec![(5, [1, 2, 3]), (6, [4, 5, 6])]
        );
        const ALL_ENTRIES: usize = 256;
        assert_eq!(
            change(0, 0, &[[9, 9, 9, 0]; ALL_ENTRIES]).len(),
            ALL_ENTRIES
        );
        assert!(change(200, 100, &[[1, 1, 1, 0]; 100]).is_empty());
    }

    #[test]
    fn a_cut_index_flags_the_entries_it_holds() {
        let bytes = write_av();
        let idx1 = find_chunk_body(&bytes, &IDX1_FOURCC).expect("the idx1");
        let file = parse(&bytes[..idx1 + IDX1_ENTRY_LEN + IDX1_ENTRY_FLAGS])
            .expect("a cut index must not fail the parse");
        assert_eq!(
            file.chunks.len(),
            parse(&bytes).expect("parses").chunks.len()
        );
        let flagged: Vec<bool> = file.chunks.iter().map(|c| c.keyframe).collect();
        assert!(flagged[0], "the one whole entry flags its chunk");
        assert!(!flagged[1..].contains(&true), "no entry, no flag");
    }

    #[test]
    fn rejects_an_absurd_stream_count() {
        let mut bytes = write_av();
        let avih = find_chunk_body(&bytes, &AVIH_FOURCC).expect("the avih");
        bytes[avih + AVIH_STREAM_COUNT..avih + AVIH_STREAM_COUNT + 4]
            .copy_from_slice(&1_000_000u32.to_le_bytes());
        assert!(parse(&bytes).is_err(), "a million streams must fail");
    }

    #[test]
    fn rejects_a_chunk_longer_than_the_file() {
        let mut bytes = write_av();
        let hdrl = find_chunk_body(&bytes, &AVIH_FOURCC).expect("the avih");
        bytes[hdrl - 4..hdrl].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(parse(&bytes).is_err(), "an oversized chunk must fail");
    }

    #[test]
    fn an_idx1_offset_past_the_file_costs_no_flags() {
        let mut bytes = write_av();
        let idx1 = find_chunk_body(&bytes, &IDX1_FOURCC).expect("the idx1");
        for entry in 0..4 {
            let at = idx1 + entry * IDX1_ENTRY_LEN + IDX1_ENTRY_OFFSET;
            bytes[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        }
        let file = parse(&bytes).expect("a bogus index must not fail the parse");
        assert_eq!(file.chunks.iter().filter(|c| c.stream == 0).count(), 4);
    }

    #[test]
    fn a_large_index_flags_the_right_chunks() {
        const CHUNKS: usize = 50_000;
        const KEYFRAME_INTERVAL: usize = 7;
        let mut writer = AviWriter::new(vec![AviWriteStream::Video {
            codec: VideoCodec::H264,
            width: 16,
            height: 16,
        }]);
        for i in 0..CHUNKS {
            writer
                .push(
                    0,
                    vec![1; 1 + i % 3],
                    i as u64 * DEFAULT_FRAME_PERIOD_NS,
                    i % KEYFRAME_INTERVAL == 0,
                )
                .expect("video chunk");
        }
        let file = parse(&writer.finish().expect("serializes")).expect("parses");
        assert_eq!(file.chunks.len(), CHUNKS);
        for (i, chunk) in file.chunks.iter().enumerate() {
            assert_eq!(chunk.keyframe, i % KEYFRAME_INTERVAL == 0, "chunk {i}");
        }
    }

    /// Offset of the body of the first chunk with `id`, by scanning: the tests
    /// patch header fields the parser reads, so they need to find them first.
    fn find_chunk_body(data: &[u8], id: &FourCc) -> Option<usize> {
        data.windows(id.len())
            .position(|w| w == id)
            .map(|at| at + CHUNK_HEADER_LEN)
    }
}
