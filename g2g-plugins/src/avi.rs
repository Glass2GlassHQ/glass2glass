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
//! a size that overruns its list, a stream count past what AVI can name, an
//! `idx1` entry pointing outside `movi`, and a truncated header all fail the
//! parse rather than panicking or allocating on the declared length.
//!
//! AVI stores chunks in decode order and carries no presentation timestamps, so
//! a stream's timing is reconstructed from `strh`: sample `n` starts at
//! `n * dwScale / dwRate` seconds, where `n` counts chunks when `dwSampleSize`
//! is 0 and `bytes / dwSampleSize` otherwise.

use alloc::vec::Vec;

use g2g_core::{AudioFormat, G2gError, Tag, TagList, VideoCodec};

use crate::riff::{
    chunks, padded_len, read_fourcc, read_u16, read_u32, FourCc, CHUNK_HEADER_LEN, LIST_FOURCC,
    RIFF_FOURCC, RIFF_HEADER_LEN,
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

/// `WAVEFORMATEX` `wFormatTag` values this maps.
const WAVE_FORMAT_PCM: u16 = 0x0001;
const WAVE_FORMAT_MP3: u16 = 0x0055;
const WAVE_FORMAT_AAC: u16 = 0x00FF;
const WAVE_FORMAT_AC3: u16 = 0x2000;

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
    Audio {
        format: AudioFormat,
        channels: u8,
        sample_rate: u32,
    },
    Unsupported,
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

/// The g2g format a `(wFormatTag, wBitsPerSample)` pair names, or `None` for a
/// payload g2g has no decoder for (ADPCM, WMA, a PCM width off the byte grid).
fn audio_format(tag: u16, bits: u16) -> Option<AudioFormat> {
    match (tag, bits) {
        (WAVE_FORMAT_PCM, 8) => Some(AudioFormat::PcmU8),
        (WAVE_FORMAT_PCM, 16) => Some(AudioFormat::PcmS16Le),
        (WAVE_FORMAT_PCM, 24) => Some(AudioFormat::PcmS24Le),
        (WAVE_FORMAT_PCM, 32) => Some(AudioFormat::PcmS32Le),
        (WAVE_FORMAT_MP3, _) => Some(AudioFormat::Mp3),
        (WAVE_FORMAT_AC3, _) => Some(AudioFormat::Ac3),
        (WAVE_FORMAT_AAC, _) => Some(AudioFormat::Aac),
        _ => None,
    }
}

/// The `wFormatTag` and bit depth to write for a g2g audio format, or `None`
/// for one AVI cannot carry.
fn wave_format_tag(format: AudioFormat) -> Option<(u16, u16)> {
    match format {
        AudioFormat::PcmU8 => Some((WAVE_FORMAT_PCM, 8)),
        AudioFormat::PcmS16Le => Some((WAVE_FORMAT_PCM, 16)),
        AudioFormat::PcmS24Le => Some((WAVE_FORMAT_PCM, 24)),
        AudioFormat::PcmS32Le => Some((WAVE_FORMAT_PCM, 32)),
        AudioFormat::Mp3 => Some((WAVE_FORMAT_MP3, 0)),
        AudioFormat::Ac3 => Some((WAVE_FORMAT_AC3, 0)),
        AudioFormat::Aac => Some((WAVE_FORMAT_AAC, 0)),
        _ => None,
    }
}

/// The `biCompression` to write for a g2g video codec, or `None` for one AVI
/// cannot carry.
fn video_fourcc(codec: VideoCodec) -> Option<FourCc> {
    VIDEO_FOURCCS
        .iter()
        .find(|(_, c)| *c == codec)
        .map(|(fourcc, _)| *fourcc)
}

/// Bytes one PCM sample frame occupies, `None` for a compressed format.
fn pcm_block_align(format: AudioFormat, channels: u8) -> Option<usize> {
    let bits = match format {
        AudioFormat::PcmU8 => 8usize,
        AudioFormat::PcmS16Le => 16,
        AudioFormat::PcmS24Le => 24,
        AudioFormat::PcmS32Le => 32,
        _ => return None,
    };
    Some(bits / 8 * channels.max(1) as usize)
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
    let kind = match read_fourcc(strh, STRH_TYPE)? {
        STREAM_TYPE_VIDEO => parse_video_format(strh, strf),
        STREAM_TYPE_AUDIO => parse_audio_format(strf),
        _ => None,
    };
    let (kind, codec_config) = kind.unwrap_or((AviStreamKind::Unsupported, Vec::new()));
    Some(AviStream {
        kind,
        scale: read_u32(strh, STRH_SCALE)?,
        rate: read_u32(strh, STRH_RATE)?,
        start: read_u32(strh, STRH_START)?,
        sample_size: read_u32(strh, STRH_SAMPLE_SIZE)?,
        codec_config,
    })
}

/// A video stream's geometry and codec from its `BITMAPINFOHEADER`, plus the
/// extradata past it (H.264 parameter sets when the writer put them there).
/// `biCompression` falls back to the `strh` `fccHandler` some writers fill in
/// instead. A negative `biHeight` means a bottom-up image, which changes no
/// decoded geometry, so the magnitude is what travels.
fn parse_video_format(strh: &[u8], strf: &[u8]) -> Option<(AviStreamKind, Vec<u8>)> {
    if strf.len() < BITMAPINFO_LEN {
        return None;
    }
    let compression = read_fourcc(strf, BITMAPINFO_COMPRESSION)?;
    let codec =
        video_codec(compression).or_else(|| video_codec(read_fourcc(strh, STRH_HANDLER)?))?;
    let width = read_u32(strf, BITMAPINFO_WIDTH)?;
    let height = (read_u32(strf, BITMAPINFO_HEIGHT)? as i32).unsigned_abs();
    if width == 0 || height == 0 {
        return None;
    }
    Some((
        AviStreamKind::Video {
            codec,
            width,
            height,
        },
        Vec::from(&strf[BITMAPINFO_LEN..]),
    ))
}

/// An audio stream's format from its `WAVEFORMATEX`, plus the `cbSize` extra
/// bytes (an AAC stream's `AudioSpecificConfig`), bounded by what the chunk
/// actually holds rather than by the declared `cbSize`.
fn parse_audio_format(strf: &[u8]) -> Option<(AviStreamKind, Vec<u8>)> {
    if strf.len() < WAVEFORMAT_LEN {
        return None;
    }
    let format = audio_format(
        read_u16(strf, WAVEFORMAT_TAG)?,
        read_u16(strf, WAVEFORMAT_BITS)?,
    )?;
    let channels = read_u16(strf, WAVEFORMAT_CHANNELS)?;
    let sample_rate = read_u32(strf, WAVEFORMAT_SAMPLE_RATE)?;
    if channels == 0 || channels > u8::MAX as u16 || sample_rate == 0 {
        return None;
    }
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
/// emits them as padding and no decoder has anything to do with them.
fn collect_movi(
    data: &[u8],
    range: core::ops::Range<usize>,
    stream_count: usize,
    out: &mut Vec<AviChunk>,
) -> Result<(), G2gError> {
    let mut walk = chunks(data, range);
    for chunk in walk.by_ref() {
        if chunk.list_form(data) == Some(REC_FOURCC) {
            collect_movi(data, chunk.list_body(), stream_count, out)?;
            continue;
        }
        let Some(stream) = chunk_stream_number(chunk.id) else {
            continue;
        };
        if stream >= stream_count || chunk.body.is_empty() {
            continue;
        }
        out.push(AviChunk {
            stream,
            body: chunk.body,
            keyframe: false,
        });
    }
    if walk.overran() {
        return Err(G2gError::CapsMismatch);
    }
    Ok(())
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
        if chunks_out.iter().any(|c| c.body.start == relative) {
            return Some(movi_form_offset);
        }
        let absolute = body_of(offset)?;
        chunks_out
            .iter()
            .any(|c| c.body.start == absolute)
            .then_some(0)
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
        if let Some(chunk) = chunks_out.iter_mut().find(|c| c.body.start == body) {
            chunk.keyframe = true;
        }
    }
}

/// Parse a whole AVI file. Fails with [`G2gError::CapsMismatch`] on anything
/// malformed: a missing or truncated `RIFF AVI ` header, a `LIST` or chunk size
/// past what the file holds, a stream count AVI cannot name, or a file with no
/// stream g2g can carry.
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
        let riff = match chunks(data, riff_at..data.len())
            .next()
            .filter(|c| c.id == RIFF_FOURCC)
        {
            Some(riff) => riff,
            // A first list whose declared size runs past the file is a
            // truncated download; anything unreadable after the last one is
            // trailing padding.
            None if riff_at == 0 => return Err(G2gError::CapsMismatch),
            None => break,
        };
        if riff_at != 0 && read_fourcc(data, riff.body.start) != Some(AVIX_FOURCC) {
            break;
        }
        let mut walk = chunks(data, riff.list_body());
        for chunk in walk.by_ref() {
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
        if walk.overran() {
            return Err(G2gError::CapsMismatch);
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
    let stream_count = file.streams.len();
    for (_, range) in &movi_ranges {
        collect_movi(data, range.clone(), stream_count, &mut file.chunks)?;
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
                }
            );
        }
    }
    Ok(file)
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
                    kind: AviStreamKind::Unsupported,
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
            Self::Audio { format, .. } => wave_format_tag(*format).is_some(),
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
        hdrl.extend_from_slice(&riff_chunk(&AVIH_FOURCC, &self.avih()));
        for stream in 0..self.streams.len() {
            let mut strl = Vec::from(STRL_FOURCC);
            strl.extend_from_slice(&riff_chunk(&STRH_FOURCC, &self.strh(stream)?));
            strl.extend_from_slice(&riff_chunk(&STRF_FOURCC, &self.strf(stream)?));
            hdrl.extend_from_slice(&riff_chunk(&LIST_FOURCC, &strl));
        }

        let mut movi = Vec::from(MOVI_FOURCC);
        let mut idx1 = Vec::with_capacity(self.chunks.len() * IDX1_ENTRY_LEN);
        for chunk in &self.chunks {
            let video = matches!(self.streams[chunk.stream], AviWriteStream::Video { .. });
            let id = chunk_id(chunk.stream, video);
            // `idx1` measures from the `movi` form type, which is where this
            // buffer starts, so the running length is the offset to record.
            idx1.extend_from_slice(&id);
            idx1.extend_from_slice(
                &(if chunk.keyframe { AVIIF_KEYFRAME } else { 0 }).to_le_bytes(),
            );
            idx1.extend_from_slice(&(movi.len() as u32).to_le_bytes());
            idx1.extend_from_slice(&(chunk.data.len() as u32).to_le_bytes());
            movi.extend_from_slice(&riff_chunk(&id, &chunk.data));
        }

        let mut body = Vec::from(AVI_FOURCC);
        body.extend_from_slice(&riff_chunk(&LIST_FOURCC, &hdrl));
        body.extend_from_slice(&riff_chunk(&LIST_FOURCC, &movi));
        body.extend_from_slice(&riff_chunk(&IDX1_FOURCC, &idx1));
        Ok(riff_chunk(&RIFF_FOURCC, &body))
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
                let (scale, size) = match pcm_block_align(*format, *channels) {
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
                let pcm_align = pcm_block_align(*format, *channels);
                let align = pcm_align.unwrap_or(1);
                let byte_rate = match pcm_align {
                    Some(align) => *sample_rate as usize * align,
                    // A compressed stream's average rate is what it actually
                    // wrote over the time its chunks covered.
                    None => self.compressed_byte_rate(stream),
                };
                out.extend_from_slice(&tag.to_le_bytes());
                out.extend_from_slice(&(*channels as u16).to_le_bytes());
                out.extend_from_slice(&sample_rate.to_le_bytes());
                out.extend_from_slice(&(byte_rate.min(u32::MAX as usize) as u32).to_le_bytes());
                out.extend_from_slice(&(align.min(u16::MAX as usize) as u16).to_le_bytes());
                out.extend_from_slice(&bits.to_le_bytes());
                out.extend_from_slice(&0u16.to_le_bytes()); // cbSize
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
    let divisor = gcd(period, NANOS_PER_SECOND);
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

fn gcd(a: u64, b: u64) -> u64 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a.max(1)
}

/// A RIFF chunk: id, little-endian size, body, and the pad byte an odd size
/// needs.
fn riff_chunk(id: &FourCc, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(CHUNK_HEADER_LEN + body.len() + 1);
    out.extend_from_slice(id);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(body);
    if body.len() % 2 == 1 {
        out.push(0);
    }
    out
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
    fn rejects_a_truncated_file() {
        let bytes = write_av();
        for cut in [0, 4, 11, 20, bytes.len() / 2, bytes.len() - 1] {
            assert!(
                parse(&bytes[..cut]).is_err(),
                "a file cut to {cut} bytes must fail the parse"
            );
        }
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

    /// Offset of the body of the first chunk with `id`, by scanning: the tests
    /// patch header fields the parser reads, so they need to find them first.
    fn find_chunk_body(data: &[u8], id: &FourCc) -> Option<usize> {
        data.windows(id.len())
            .position(|w| w == id)
            .map(|at| at + CHUNK_HEADER_LEN)
    }
}
