//! Matroska / WebM demuxer (M110): parse an EBML byte stream into the
//! elementary-stream frames its Clusters carry (RFC 9559 / the matroska.org
//! spec).
//!
//! Pure `no_std + alloc` parsing, the [`crate::mpegts`] precedent for the MKV
//! container: this module is the state machine (read EBML elements, descend into
//! the Segment, read Tracks for the elementary streams, read each Cluster's
//! SimpleBlocks into frames with timestamps). The [`crate::mkvdemux::MkvDemux`]
//! element wraps it; the split keeps the bit-twiddling testable without a runner.
//!
//! EBML basics: every element is `(id, size, body)`. The id is a 1..4 byte
//! variable-length integer kept whole (the length marker is part of the value);
//! the size is a variable-length integer with its marker stripped, or all-ones
//! for "unknown size". Master elements (Segment, Tracks, Cluster, ...) nest
//! children in their body.
//!
//! Scope (v1): a single Segment. Clusters, definite-size and unknown-size (the
//! live-streaming shape) alike, are descended into and their children parsed one
//! at a time until the next top-level element ends them, so the blocks of a
//! Cluster a truncated file cuts short still demux. A Cluster child that names no
//! declared track or carries no usable size is damage: the parser drops bytes up
//! to the next Cluster ID and carries on there. SimpleBlock / Block frames,
//! including all three lacing modes (Xiph / EBML / fixed), are split; laced frames
//! share the block timestamp. The `Cues` index is parsed into a time -> Cluster
//! byte-position map ([`MatroskaDemuxer::cue_seek_offset`]) for indexed seeking
//! (M373). BlockGroup reference tracking and per-frame timestamp interpolation
//! from DefaultDuration are follow-ups.

use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::log::Target;
#[cfg(feature = "ffmpeg-long-tail")]
use g2g_core::TagNamespace;
use g2g_core::{
    g2g_warn, AudioFormat, Chapter, CodecTag, ProResVariant, RawVideoFormat, SubPictureFormat, Tag,
    TagList, TextFormat,
};

use crate::dvbsub::{page_id_blob, parse_page_ids, segment_span, PageIds, DEFAULT_SUBTITLING_TYPE};
use crate::fifobuffer::FifoBuffer;
use crate::subparse::ASS_SCRIPT_HEADER;
use crate::vobsub::{idx_config_text, parse_idx};

/// EBML / Matroska element IDs (kept whole, length marker included). The demuxer
/// skips the EBML header by its size and ignores TrackType (the CodecID pins the
/// media type), but the muxer writes both, so they are named here too.
const ID_EBML: u32 = 0x1A45_DFA3;
const ID_SEGMENT: u32 = 0x1853_8067;
const ID_INFO: u32 = 0x1549_A966;
const ID_TIMESTAMP_SCALE: u32 = 0x002A_D7B1;
const ID_DURATION: u32 = 0x4489;
const ID_TITLE: u32 = 0x7BA9;
const ID_MUXING_APP: u32 = 0x4D80;
const ID_WRITING_APP: u32 = 0x5741;
// CodecDelay and SeekPreRoll are v4 elements, so the file says so like ffmpeg does
const DOC_TYPE_VERSION: u8 = 4;
const DOC_TYPE_READ_VERSION: u8 = 2;
const APP_NAME: &str = concat!("g2g-plugins ", env!("CARGO_PKG_VERSION"));
const ID_TAGS: u32 = 0x1254_C367;
const ID_TAG: u32 = 0x7373;
const ID_TARGETS: u32 = 0x63C0;
const ID_TAG_TRACK_UID: u32 = 0x63C5;
const ID_SIMPLE_TAG: u32 = 0x67C8;
const ID_TAG_NAME: u32 = 0x45A3;
const ID_TAG_STRING: u32 = 0x4487;
const ID_TRACKS: u32 = 0x1654_AE6B;
const ID_TRACK_ENTRY: u32 = 0x00AE;
const ID_TRACK_NUMBER: u32 = 0x00D7;
const ID_TRACK_UID: u32 = 0x73C5;
const ID_TRACK_NAME: u32 = 0x536E;
const ID_CODEC_DELAY: u32 = 0x56AA;
const ID_SEEK_PRE_ROLL: u32 = 0x56BB;
const ID_LANGUAGE: u32 = 0x0022_B59C;
const ID_LANGUAGE_BCP47: u32 = 0x0022_B59D;
const ID_TRACK_TYPE: u32 = 0x0083;
const ID_CODEC_ID: u32 = 0x0086;
const ID_DEFAULT_DURATION: u32 = 0x0023_E383;
const ID_VIDEO: u32 = 0x00E0;
const ID_PIXEL_WIDTH: u32 = 0x00B0;
const ID_PIXEL_HEIGHT: u32 = 0x00BA;
const ID_COLOUR_SPACE: u32 = 0x002E_B524;
const ID_AUDIO: u32 = 0x00E1;
const ID_CONTENT_ENCODINGS: u32 = 0x6D80;
const ID_CONTENT_ENCODING: u32 = 0x6240;
const ID_CONTENT_ENCODING_ORDER: u32 = 0x5031;
const ID_CONTENT_ENCODING_SCOPE: u32 = 0x5032;
const ID_CONTENT_ENCODING_TYPE: u32 = 0x5033;
const ID_CONTENT_COMPRESSION: u32 = 0x5034;
const ID_CONTENT_COMP_ALGO: u32 = 0x4254;
const ID_CONTENT_COMP_SETTINGS: u32 = 0x4255;
const ID_CONTENT_ENCRYPTION: u32 = 0x5035;
const ID_CONTENT_ENC_ALGO: u32 = 0x47E1;
const ID_CONTENT_ENC_KEY_ID: u32 = 0x47E2;
const ID_CONTENT_ENC_AES_SETTINGS: u32 = 0x47E7;
const ID_AES_SETTINGS_CIPHER_MODE: u32 = 0x47E8;
const ID_CHANNELS: u32 = 0x009F;
const ID_SAMPLING_FREQ: u32 = 0x00B5;
const ID_BIT_DEPTH: u32 = 0x6264;
const ID_CLUSTER: u32 = 0x1F43_B675;
const ID_TIMESTAMP: u32 = 0x00E7;
const ID_SIMPLE_BLOCK: u32 = 0x00A3;
const ID_BLOCK_GROUP: u32 = 0x00A0;
const ID_BLOCK: u32 = 0x00A1;
const ID_BLOCK_DURATION: u32 = 0x009B;
const ID_DISCARD_PADDING: u32 = 0x75A2;
const ID_SEEK_HEAD: u32 = 0x114D_9B74;
const ID_SEEK: u32 = 0x4DBB;
const ID_SEEK_ID: u32 = 0x53AB;
const ID_SEEK_POSITION: u32 = 0x53AC;
const ID_CUES: u32 = 0x1C53_BB6B;
const ID_CUE_POINT: u32 = 0x00BB;
const ID_CUE_TIME: u32 = 0x00B3;
const ID_CUE_TRACK_POSITIONS: u32 = 0x00B7;
const ID_CUE_TRACK: u32 = 0x00F7;
const ID_CUE_CLUSTER_POSITION: u32 = 0x00F1;
const ID_CHAPTERS: u32 = 0x1043_A770;
const ID_EDITION_ENTRY: u32 = 0x45B9;
const ID_EDITION_FLAG_HIDDEN: u32 = 0x45BD;
const ID_EDITION_FLAG_DEFAULT: u32 = 0x45DB;
const ID_CHAPTER_ATOM: u32 = 0x00B6;
const ID_CHAPTER_UID: u32 = 0x73C4;
const ID_CHAPTER_TIME_START: u32 = 0x0091;
const ID_CHAPTER_TIME_END: u32 = 0x0092;
const ID_CHAPTER_FLAG_HIDDEN: u32 = 0x0098;
const ID_CHAPTER_DISPLAY: u32 = 0x0080;
const ID_CHAP_STRING: u32 = 0x0085;
const ID_CHAP_LANGUAGE: u32 = 0x437C;
const ID_ATTACHMENTS: u32 = 0x1941_A469;

/// Whether `id` is a Segment-level (level-1) element. Used to decide when a
/// Cluster ends: only a real level-1 sibling closes it, so a benign Cluster
/// child (Void, CRC-32, Position, PrevSize) is skipped instead.
fn is_level1_element(id: u32) -> bool {
    matches!(
        id,
        ID_EBML
            | ID_SEGMENT
            | ID_SEEK_HEAD
            | ID_INFO
            | ID_TRACKS
            | ID_CUES
            | ID_CHAPTERS
            | ID_TAGS
            | ID_ATTACHMENTS
            | ID_CLUSTER
    )
}

/// The default `TimestampScale` (ns per timestamp unit) when `Info` omits it.
const DEFAULT_TIMESTAMP_SCALE: u64 = 1_000_000;

/// The codec a Matroska track carries, mapped from its `CodecID` string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MkvCodec {
    H264,
    H265,
    Vp8,
    Vp9,
    Av1,
    // `V_MJPEG`, or `V_MS/VFW/FOURCC` whose BITMAPINFOHEADER names `MJPG`
    Mjpeg,
    Aac,
    Opus,
    /// Dolby Digital audio (`A_AC3`, incl. BSID-suffixed forms). Self-syncing
    /// frames, no CodecPrivate needed.
    Ac3,
    /// FLAC audio (`A_FLAC`). The `CodecPrivate` carries the native `fLaC`
    /// STREAMINFO header the decoder needs as extradata.
    Flac,
    // `A_VORBIS`, the three header packets Xiph-laced in the `CodecPrivate`
    Vorbis,
    /// A timed-text subtitle track (`S_TEXT/*`); `format` names the on-disk block
    /// payload's syntax, which the demuxer de-frames to plain UTF-8 cue text:
    /// `S_TEXT/UTF8` -> [`TextFormat::Utf8`] (verbatim), `S_TEXT/ASS` / `S_TEXT/SSA`
    /// -> [`TextFormat::Ssa`] (the `Text` field of the comma-separated block, tags
    /// stripped), `S_TEXT/WEBVTT` -> [`TextFormat::WebVtt`] (cue text, inline tags
    /// stripped). The bitmap subtitle codecs are not text: `S_VOBSUB`,
    /// `S_DVBSUB` and `S_HDMV/PGS` have their own variants.
    Subtitle(TextFormat),
    /// A DVD subpicture (bitmap) subtitle track (`S_VOBSUB`). Each block is one
    /// SPU packet; the track's `CodecPrivate` is the `.idx` text carrying the
    /// palette and display size the decoder needs.
    VobSub,
    /// A DVB subtitle (bitmap) track (`S_DVBSUB`). Each block is one display
    /// set's segments, without the PES data-field header; the track's
    /// `CodecPrivate` carries the composition and ancillary page ids.
    DvbSub,
    /// A Blu-ray PGS subtitle (bitmap) track (`S_HDMV/PGS`). Each block is one
    /// display set's segments, without the `.sup` per-segment `PG` / PTS / DTS
    /// header; the track carries no `CodecPrivate`.
    Pgs,
    /// A `CodecID` this demuxer does not map to a g2g caps type.
    Other,
    /// Raw PCM (`A_PCM/INT/LIT`, `A_PCM/INT/BIG`, `A_PCM/FLOAT/IEEE`): `format` is
    /// the CodecID's layout at the track's `BitDepth`, and `big_endian` samples are
    /// byte-swapped into it at demux.
    Pcm {
        format: AudioFormat,
        big_endian: bool,
    },
    // `V_UNCOMPRESSED` with its `ColourSpace` fourcc, or a `V_MS/VFW/FOURCC` naming a raw layout
    RawVideo(RawVideoFormat),
    // `V_THEORA`, the three header packets Xiph-laced in the `CodecPrivate`
    Theora,
    // `V_PRORES`, its profile's fourcc in the `CodecPrivate`
    ProRes(ProResVariant),
    // `V_MS/VFW/FOURCC` or `V_QUICKTIME` naming a codec no variant above maps, its
    // `CodecPrivate` the BITMAPINFOHEADER or the sample entry
    TaggedVideo(CodecTag),
    // `A_MS/ACM` naming a codec no variant above maps, its `CodecPrivate` the WAVEFORMATEX
    TaggedAudio(CodecTag),
}

impl MkvCodec {
    /// Map a Matroska `CodecID` string to a codec. AAC has profile suffixes
    /// (`A_AAC/MPEG4/LC`, ...), so it is matched by prefix; the rest are exact.
    fn from_codec_id(id: &[u8]) -> MkvCodec {
        // EBML strings may be NUL-padded (GStreamer's matroskamux pads CodecID)
        let padding = id.iter().rev().take_while(|&&byte| byte == 0).count();
        let id = &id[..id.len() - padding];
        if id == b"V_MPEG4/ISO/AVC" {
            MkvCodec::H264
        } else if id == b"V_MPEGH/ISO/HEVC" {
            MkvCodec::H265
        } else if id == b"V_VP8" {
            MkvCodec::Vp8
        } else if id == b"V_VP9" {
            MkvCodec::Vp9
        } else if id == b"V_AV1" {
            MkvCodec::Av1
        } else if id == b"V_MJPEG" {
            MkvCodec::Mjpeg
        } else if id == b"V_THEORA" {
            MkvCodec::Theora
        } else if id == b"A_VORBIS" {
            MkvCodec::Vorbis
        } else if id.starts_with(b"A_AAC") {
            MkvCodec::Aac
        } else if id == b"A_OPUS" {
            MkvCodec::Opus
        } else if id.starts_with(b"A_AC3") {
            MkvCodec::Ac3
        } else if id == b"A_FLAC" {
            MkvCodec::Flac
        } else if id == b"S_TEXT/UTF8" {
            MkvCodec::Subtitle(TextFormat::Utf8)
        } else if id == b"S_TEXT/ASS" || id == b"S_TEXT/SSA" {
            MkvCodec::Subtitle(TextFormat::Ssa)
        } else if id == b"S_TEXT/WEBVTT" {
            MkvCodec::Subtitle(TextFormat::WebVtt)
        } else if id == b"S_VOBSUB" {
            MkvCodec::VobSub
        } else if id == b"S_DVBSUB" {
            MkvCodec::DvbSub
        } else if id == b"S_HDMV/PGS" {
            MkvCodec::Pgs
        } else {
            MkvCodec::Other
        }
    }

    /// The canonical Matroska `CodecID` to write for a codec (`None` for the
    /// unmappable [`MkvCodec::Other`]). AAC writes the LC profile string.
    pub fn codec_id(self) -> Option<&'static [u8]> {
        Some(match self {
            MkvCodec::H264 => b"V_MPEG4/ISO/AVC",
            MkvCodec::H265 => b"V_MPEGH/ISO/HEVC",
            MkvCodec::Vp8 => b"V_VP8",
            MkvCodec::Vp9 => b"V_VP9",
            MkvCodec::Av1 => b"V_AV1",
            MkvCodec::Mjpeg => b"V_MJPEG",
            MkvCodec::Vorbis => b"A_VORBIS",
            MkvCodec::Aac => b"A_AAC",
            MkvCodec::Opus => b"A_OPUS",
            MkvCodec::Ac3 => b"A_AC3",
            MkvCodec::Flac => b"A_FLAC",
            MkvCodec::Subtitle(TextFormat::Utf8) => b"S_TEXT/UTF8",
            MkvCodec::Subtitle(TextFormat::Ssa) => b"S_TEXT/ASS",
            // No `S_TEXT/WEBVTT`: it is a read-side mapping only. ffmpeg writes and
            // reads the WebM `D_WEBVTT/*` ids instead, whose block payload leads
            // with the cue identifier and settings, so writing WebVTT here would be
            // a carriage no reference peer reads back.
            MkvCodec::VobSub => b"S_VOBSUB",
            MkvCodec::DvbSub => b"S_DVBSUB",
            MkvCodec::Pgs => b"S_HDMV/PGS",
            // `A_PCM/*` is a read-side mapping only: the muxer writes no `BitDepth`.
            MkvCodec::Subtitle(_)
            | MkvCodec::Other
            | MkvCodec::Pcm { .. }
            | MkvCodec::RawVideo(_)
            | MkvCodec::Theora
            | MkvCodec::ProRes(_)
            | MkvCodec::TaggedVideo(_)
            | MkvCodec::TaggedAudio(_) => return None,
        })
    }

    /// True for the WebM codec subset, so the muxer can write the `webm` DocType.
    pub fn is_webm(self) -> bool {
        matches!(
            self,
            MkvCodec::Vp8 | MkvCodec::Vp9 | MkvCodec::Av1 | MkvCodec::Opus | MkvCodec::Vorbis
        )
    }

    /// `1` for video, `2` for audio, `0x11` for subtitle (the Matroska `TrackType`).
    fn track_type(self) -> u8 {
        match self {
            MkvCodec::Aac
            | MkvCodec::Opus
            | MkvCodec::Ac3
            | MkvCodec::Flac
            | MkvCodec::Vorbis
            | MkvCodec::Pcm { .. }
            | MkvCodec::TaggedAudio(_) => 2,
            MkvCodec::Subtitle(_) | MkvCodec::VobSub | MkvCodec::DvbSub | MkvCodec::Pgs => 0x11,
            _ => 1,
        }
    }
}

/// Cap on one inflated block (M910). A Matroska zlib track is a subtitle or
/// metadata carrier in practice, so a real block is kilobytes; the bound is what
/// stops a crafted few-KiB block that inflates to gigabytes.
const MAX_INFLATED_BLOCK_LEN: usize = 16 * 1024 * 1024;

// a real CodecPrivate is decoder setup (an avcC, Vorbis headers, an ASS script header) of tens of KiB, held for the whole stream
const MAX_INFLATED_CODEC_PRIVATE_LEN: usize = 1024 * 1024;

/// Cap on the `ContentCompSettings` bytes a header-stripped track prepends back
/// onto every block. Real strippings are a handful of bytes (an MPEG-4 start
/// code, an AAC ADTS header); a longer one is the file's claim, not a header.
const MAX_HEADER_STRIP_LEN: usize = 256;

// every step is undone on every block, and ffmpeg reads no more than one
const MAX_CONTENT_ENCODINGS: usize = 8;

const SCOPE_BLOCK: u64 = 1;
const SCOPE_CODEC_PRIVATE: u64 = 2;

const TYPE_COMPRESSION: u64 = 0;
const TYPE_ENCRYPTION: u64 = 1;
const CONTENT_ENC_ALGO_AES: u64 = 5;
const AES_CIPHER_MODE_CTR: u64 = 1;

// one `ContentCompAlgo` this demuxer undoes
#[derive(Debug, Clone, PartialEq, Eq)]
enum ContentCompression {
    Zlib,
    #[cfg(feature = "matroska-compression")]
    Bzip2,
    #[cfg(feature = "matroska-compression")]
    Lzo,
    // these bytes were stripped from the head of the data and are prepended back
    HeaderStrip(Vec<u8>),
}

impl ContentCompression {
    // `None` when the bytes do not decode or the output exceeds `limit`
    fn decode(&self, stored: &[u8], limit: usize) -> Option<Vec<u8>> {
        match self {
            ContentCompression::Zlib => {
                miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(stored, limit).ok()
            }
            #[cfg(feature = "matroska-compression")]
            ContentCompression::Bzip2 => bunzip(stored, limit),
            #[cfg(feature = "matroska-compression")]
            ContentCompression::Lzo => unlzo(stored, limit),
            ContentCompression::HeaderStrip(header) => {
                let total = header.len().checked_add(stored.len())?;
                (total <= limit).then(|| [header.as_slice(), stored].concat())
            }
        }
    }
}

// one `ContentEncoding` step this demuxer undoes
#[derive(Debug, Clone)]
enum ContentDecoding {
    Decompress(ContentCompression),
    #[cfg(feature = "matroska-decrypt")]
    Decrypt(AesKey),
}

impl ContentDecoding {
    fn undo(&self, stored: &[u8], limit: usize) -> Option<Vec<u8>> {
        match self {
            ContentDecoding::Decompress(compression) => compression.decode(stored, limit),
            #[cfg(feature = "matroska-decrypt")]
            ContentDecoding::Decrypt(key) => decrypt_webm_frame(stored, key),
        }
    }
}

#[cfg(feature = "matroska-decrypt")]
#[derive(Clone, Copy)]
struct AesKey([u8; 16]);

// the key bytes never reach a log
#[cfg(feature = "matroska-decrypt")]
impl core::fmt::Debug for AesKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("AesKey(..)")
    }
}

// bits of the signal byte that leads every frame of a WebM Encryption track
#[cfg(feature = "matroska-decrypt")]
const SIGNAL_ENCRYPTED: u8 = 0x01;
#[cfg(feature = "matroska-decrypt")]
const SIGNAL_PARTITIONED: u8 = 0x02;
// a second signal byte follows, which no version of the format defines yet
#[cfg(feature = "matroska-decrypt")]
const SIGNAL_EXTENDED: u8 = 0x80;
#[cfg(feature = "matroska-decrypt")]
const WEBM_IV_LEN: usize = 8;
#[cfg(feature = "matroska-decrypt")]
const PARTITION_OFFSET_LEN: usize = 4;

// `None` when the signal byte, the IV or the partition table is malformed
#[cfg(feature = "matroska-decrypt")]
fn decrypt_webm_frame(stored: &[u8], key: &AesKey) -> Option<Vec<u8>> {
    let (&signal, rest) = stored.split_first()?;
    if signal & SIGNAL_EXTENDED != 0 {
        return None;
    }
    if signal & SIGNAL_ENCRYPTED == 0 {
        // only an encrypted frame can be partitioned
        return (signal & SIGNAL_PARTITIONED == 0).then(|| rest.to_vec());
    }
    let (iv, rest) = rest.split_first_chunk::<WEBM_IV_LEN>()?;
    let mut counter_block = [0u8; 16];
    counter_block[..WEBM_IV_LEN].copy_from_slice(iv);
    if signal & SIGNAL_PARTITIONED == 0 {
        let mut frame = rest.to_vec();
        let whole = (0, frame.len());
        crate::aes_ctr::apply_aes128_ctr(&mut frame, [whole], &key.0, &counter_block);
        return Some(frame);
    }
    let (&partition_count, rest) = rest.split_first()?;
    // the u8 count bounds the table at 1020 bytes
    let (partition_table, frame) =
        rest.split_at_checked(usize::from(partition_count) * PARTITION_OFFSET_LEN)?;
    let sections = encrypted_sections(partition_table, frame.len())?;
    let mut frame = frame.to_vec();
    crate::aes_ctr::apply_aes128_ctr(&mut frame, sections, &key.0, &counter_block);
    Some(frame)
}

// the offsets cut the frame into clear and encrypted sections in turn, a clear one first
#[cfg(feature = "matroska-decrypt")]
fn encrypted_sections(
    partition_table: &[u8],
    frame_len: usize,
) -> Option<impl Iterator<Item = (usize, usize)> + '_> {
    let bounds = move || {
        let offsets = partition_table
            .chunks_exact(PARTITION_OFFSET_LEN)
            .map(|offset| {
                u32::from_be_bytes([offset[0], offset[1], offset[2], offset[3]]) as usize
            });
        core::iter::once(0)
            .chain(offsets)
            .chain(core::iter::once(frame_len))
    };
    let sections = move || bounds().zip(bounds().skip(1));
    if sections().any(|(start, end)| end < start) {
        return None;
    }
    Some(sections().skip(1).step_by(2))
}

// the next output buffer size: three times the last, as ffmpeg's matroskadec grows it
#[cfg(feature = "matroska-compression")]
fn grown_output_len(len: usize, limit: usize) -> usize {
    const FIRST_OUTPUT_LEN: usize = 4096;
    len.saturating_mul(3).max(FIRST_OUTPUT_LEN).min(limit)
}

#[cfg(feature = "matroska-compression")]
fn bunzip(stored: &[u8], limit: usize) -> Option<Vec<u8>> {
    use libbz2_rs_sys::{bz_stream, BZ2_bzDecompressEnd, BZ2_bzDecompressInit, BZ_OK};
    let mut stream = bz_stream {
        next_in: stored.as_ptr().cast(),
        avail_in: core::ffi::c_uint::try_from(stored.len()).ok()?,
        total_in_lo32: 0,
        total_in_hi32: 0,
        next_out: core::ptr::null_mut(),
        avail_out: 0,
        total_out_lo32: 0,
        total_out_hi32: 0,
        state: core::ptr::null_mut(),
        bzalloc: None,
        bzfree: None,
        opaque: core::ptr::null_mut(),
    };
    // SAFETY: `stream` is a valid exclusive bz_stream with no custom allocator, so
    // the library allocates its state with the Rust global allocator.
    if unsafe { BZ2_bzDecompressInit(&mut stream, 0, 0) } != BZ_OK {
        return None;
    }
    let decoded = bunzip_stream(&mut stream, stored.len(), limit);
    // SAFETY: `stream` was initialised by BZ2_bzDecompressInit above and has not
    // moved since, and it is ended exactly once.
    unsafe { BZ2_bzDecompressEnd(&mut stream) };
    decoded
}

// `stream` reads the caller's `stored_len` input bytes, still borrowed by the caller
#[cfg(feature = "matroska-compression")]
fn bunzip_stream(
    stream: &mut libbz2_rs_sys::bz_stream,
    stored_len: usize,
    limit: usize,
) -> Option<Vec<u8>> {
    use libbz2_rs_sys::{BZ2_bzDecompress, BZ_OK, BZ_STREAM_END};
    let mut out = alloc::vec![0u8; grown_output_len(stored_len, limit)];
    let mut written = 0;
    loop {
        if written == out.len() {
            if out.len() == limit {
                return None;
            }
            out.resize(grown_output_len(out.len(), limit), 0);
        }
        let room = &mut out[written..];
        stream.next_out = room.as_mut_ptr().cast();
        stream.avail_out = core::ffi::c_uint::try_from(room.len()).ok()?;
        // SAFETY: `stream` was initialised by BZ2_bzDecompressInit, `next_in`
        // points into the input the caller still borrows with `avail_in` bytes
        // left, and `next_out` points at `room`, writable for `avail_out` bytes.
        let code = unsafe { BZ2_bzDecompress(stream) };
        written = out.len() - stream.avail_out as usize;
        match code {
            BZ_STREAM_END => {
                out.truncate(written);
                return Some(out);
            }
            // room left over means the input ended before the bzip2 stream did
            BZ_OK if stream.avail_out != 0 => return None,
            BZ_OK => {}
            _ => return None,
        }
    }
}

#[cfg(feature = "matroska-compression")]
fn unlzo(stored: &[u8], limit: usize) -> Option<Vec<u8>> {
    let mut out = alloc::vec![0u8; grown_output_len(stored.len(), limit)];
    loop {
        match lzo::decompress_into(stored, &mut out) {
            Ok(len) => {
                out.truncate(len);
                return Some(out);
            }
            // raw LZO1X does not carry its decoded length
            Err(lzo::Error::OutputOverrun) if out.len() < limit => {
                out.resize(grown_output_len(out.len(), limit), 0);
            }
            Err(_) => return None,
        }
    }
}

// the steps that decode a track's blocks and its `CodecPrivate`, highest `ContentEncodingOrder` first
#[derive(Debug, Default)]
struct ContentEncodingChains {
    block: Vec<ContentDecoding>,
    codec_private: Vec<ContentDecoding>,
}

#[derive(Debug)]
struct ContentEncodingStep {
    order: u64,
    scope: u64,
    decoding: ContentDecoding,
}

// the `ContentDecoding::Decrypt` for a `ContentEncKeyID`, `None` without its key
type ContentKeyLookup<'a> = &'a dyn Fn(&[u8]) -> Option<ContentDecoding>;

fn undo_encodings(chain: &[ContentDecoding], stored: Vec<u8>, limit: usize) -> Option<Vec<u8>> {
    chain
        .iter()
        .try_fold(stored, |data, step| step.undo(&data, limit))
}

/// Read a `TrackEntry`'s `ContentEncodings` (M910). `None` when any step is one
/// this demuxer does not undo, so the track is flagged rather than its encoded
/// bytes forwarded as if they were the bitstream.
fn parse_content_encodings(
    data: &[u8],
    content_key: ContentKeyLookup<'_>,
) -> Option<ContentEncodingChains> {
    let mut steps = Vec::new();
    for (_, encoding) in children(data).filter(|(id, _)| *id == ID_CONTENT_ENCODING) {
        if steps.len() == MAX_CONTENT_ENCODINGS {
            return None;
        }
        steps.push(parse_content_encoding(encoding, content_key)?);
    }
    // RFC 9559 undoes the highest order first and requires each order to be unique
    steps.sort_unstable_by_key(|step| core::cmp::Reverse(step.order));
    if steps.windows(2).any(|pair| pair[0].order == pair[1].order) {
        return None;
    }
    let mut chains = ContentEncodingChains::default();
    for step in steps {
        if step.scope & SCOPE_BLOCK != 0 {
            chains.block.push(step.decoding.clone());
        }
        if step.scope & SCOPE_CODEC_PRIVATE != 0 {
            chains.codec_private.push(step.decoding);
        }
    }
    Some(chains)
}

fn parse_content_encoding(
    encoding: &[u8],
    content_key: ContentKeyLookup<'_>,
) -> Option<ContentEncodingStep> {
    // Defaults per the spec: order 0, scope 1 (block data), type 0 (compression).
    let mut order = 0u64;
    let mut scope = SCOPE_BLOCK;
    let mut enc_type = TYPE_COMPRESSION;
    let mut compression: Option<&[u8]> = None;
    let mut encryption: Option<&[u8]> = None;
    for (id, body) in children(encoding) {
        match id {
            ID_CONTENT_ENCODING_ORDER => order = read_uint(body),
            ID_CONTENT_ENCODING_SCOPE => scope = read_uint(body),
            ID_CONTENT_ENCODING_TYPE => enc_type = read_uint(body),
            ID_CONTENT_COMPRESSION => compression = Some(body),
            ID_CONTENT_ENCRYPTION => encryption = Some(body),
            _ => {}
        }
    }
    // scope 0 is out of range and bit 4 (the next encoding) is one players do not support
    let undoable_scope = scope != 0 && scope & !(SCOPE_BLOCK | SCOPE_CODEC_PRIVATE) == 0;
    if !undoable_scope {
        return None;
    }
    let decoding = match (enc_type, encryption) {
        // A ContentEncoding of type compression with no ContentCompression element
        // still means compression, with every default: algo 0, zlib.
        (TYPE_COMPRESSION, None) => {
            ContentDecoding::Decompress(parse_content_compression(compression.unwrap_or(&[]))?)
        }
        // WebM Encryption lays out a signal byte and IV on frames only, never on a CodecPrivate
        (TYPE_ENCRYPTION, Some(encryption)) if scope == SCOPE_BLOCK => {
            parse_content_encryption(encryption, content_key)?
        }
        _ => return None,
    };
    Some(ContentEncodingStep {
        order,
        scope,
        decoding,
    })
}

fn parse_content_compression(compression: &[u8]) -> Option<ContentCompression> {
    let mut algo = 0u64;
    let mut settings: &[u8] = &[];
    for (id, body) in children(compression) {
        match id {
            ID_CONTENT_COMP_ALGO => algo = read_uint(body),
            ID_CONTENT_COMP_SETTINGS => settings = body,
            _ => {}
        }
    }
    let compression = match algo {
        0 => ContentCompression::Zlib,
        #[cfg(feature = "matroska-compression")]
        1 => ContentCompression::Bzip2,
        #[cfg(feature = "matroska-compression")]
        2 => ContentCompression::Lzo,
        3 if settings.len() <= MAX_HEADER_STRIP_LEN => {
            ContentCompression::HeaderStrip(settings.to_vec())
        }
        // bzip2 (1) and lzo (2) without matroska-compression, an unknown algo, an absurd stripping
        _ => return None,
    };
    Some(compression)
}

// WebM Encryption's AES-CTR is the only encryption with a defined frame layout
fn parse_content_encryption(
    encryption: &[u8],
    content_key: ContentKeyLookup<'_>,
) -> Option<ContentDecoding> {
    let mut algo = 0u64;
    let mut key_id: &[u8] = &[];
    // WebM Encryption predates ContentEncAESSettings and is CTR without it
    let mut cipher_mode = AES_CIPHER_MODE_CTR;
    for (id, body) in children(encryption) {
        match id {
            ID_CONTENT_ENC_ALGO => algo = read_uint(body),
            ID_CONTENT_ENC_KEY_ID => key_id = body,
            ID_CONTENT_ENC_AES_SETTINGS => {
                for (id, body) in children(body) {
                    if id == ID_AES_SETTINGS_CIPHER_MODE {
                        cipher_mode = read_uint(body);
                    }
                }
            }
            _ => {}
        }
    }
    if algo != CONTENT_ENC_ALGO_AES || cipher_mode != AES_CIPHER_MODE_CTR {
        return None;
    }
    content_key(key_id)
}

#[derive(Debug)]
struct TrackBlockDecoding {
    track: u64,
    chain: Vec<ContentDecoding>,
    dropped_block_logged: bool,
}

/// Undo each frame's track encoding in place, dropping a block whose payload
/// does not decode (a malformed zlib stream, a malformed encryption signal byte,
/// IV or partition table), like a malformed lacing.
fn decode_encodings(frames: &mut Vec<MkvFrame>, encodings: &mut [TrackBlockDecoding]) {
    if encodings.is_empty() {
        return;
    }
    frames.retain_mut(|f| {
        let Some(decoding) = encodings.iter_mut().find(|d| d.track == f.track) else {
            return true;
        };
        match undo_encodings(
            &decoding.chain,
            core::mem::take(&mut f.data),
            MAX_INFLATED_BLOCK_LEN,
        ) {
            Some(data) => {
                f.data = data;
                true
            }
            None => {
                if !decoding.dropped_block_logged {
                    decoding.dropped_block_logged = true;
                    g2g_warn!(
                        Target::category("mkvdemux"),
                        "track {} drops blocks whose ContentEncoding does not undo",
                        f.track
                    );
                }
                false
            }
        }
    });
}

/// One elementary stream announced by a `TrackEntry`. Geometry (`width` /
/// `height`) is set for video, `channels` / `sample_rate` for audio; the others
/// stay zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MkvTrack {
    pub number: u64,
    /// The track's `TrackUID`, the id a `Tags` element's `Targets` scopes a tag
    /// to. `0` when the track omits it (then no tag can name this track).
    pub uid: u64,
    pub codec: MkvCodec,
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub sample_rate: u32,
    /// Nanoseconds per frame from `DefaultDuration` (0 when the track omits it).
    /// Spaces the frames of a laced block; an unscaled value, unlike block
    /// timestamps.
    pub default_duration_ns: u64,
    /// Whether the track declares a `ContentEncoding` this demuxer cannot undo
    /// (an encryption without its content key or without the `matroska-decrypt`
    /// feature, an unknown algorithm, bzip2 / lzo without the
    /// `matroska-compression` feature, a scope naming the next encoding). Such
    /// blocks forward as the file stored them, so a consumer of a flagged track
    /// would be reading encoded bytes as a bitstream. A chain of the other steps
    /// is undone at demux, on the blocks and the `CodecPrivate`, and never sets this.
    pub unsupported_encoding: bool,
}

/// One `CuePoint` from the Segment `Cues` index: a seekable time and the byte
/// position of the Cluster that holds it. `cluster_position` is **relative to the
/// Segment's data start** (the byte after the Segment element header), per the
/// Matroska spec; [`MatroskaDemuxer::cue_seek_offset`] adds the tracked Segment
/// data offset to give an absolute file offset for an upstream byte-seek.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CuePoint {
    /// Seekable presentation time in nanoseconds (`CueTime` scaled).
    pub time_ns: u64,
    /// `CueClusterPosition`: the Cluster's byte offset from the Segment data start.
    pub cluster_position: u64,
}

/// One demuxed frame (a SimpleBlock / Block payload) of an elementary stream.
#[derive(Debug, Clone, PartialEq)]
pub struct MkvFrame {
    pub track: u64,
    pub codec: MkvCodec,
    /// Presentation timestamp in nanoseconds (cluster + block, scaled).
    pub pts_ns: u64,
    /// The cue's display duration in nanoseconds, from the `BlockGroup`'s
    /// `BlockDuration` (scaled). `0` when the block carries none, e.g. a
    /// `SimpleBlock` (so video / audio, paced by PTS deltas, leave it `0`).
    pub duration_ns: u64,
    pub keyframe: bool,
    pub data: Vec<u8>,
}

/// Incremental Matroska demuxer: feed bytes, drain [`MkvFrame`]s.
#[derive(Debug)]
pub struct MatroskaDemuxer {
    buf: FifoBuffer<u8>,
    in_segment: bool,
    timestamp_scale: u64,
    /// The Segment `Info` `Duration`, scaled to nanoseconds. `None` until Info
    /// parses, or for a file whose Info omits it (a live / open-ended stream).
    duration_ns: Option<u64>,
    tracks: Vec<MkvTrack>,
    /// Per-track `CodecPrivate` decoder-init bytes (track number, bytes), kept
    /// beside `tracks` so `MkvTrack` stays `Copy`. Empty entries are not stored.
    codec_privates: Vec<(u64, Vec<u8>)>,
    /// Per-track `ContentEncoding` chain (track number, steps), kept beside
    /// `tracks` for the same reason as `codec_privates`. Only tracks whose blocks
    /// carry one are stored; their blocks are decoded before they are queued.
    block_encodings: Vec<TrackBlockDecoding>,
    #[cfg(feature = "matroska-decrypt")]
    content_keys: Vec<([u8; 16], AesKey)>,
    // the `CodecID` of each audio or video track whose codec is `MkvCodec::Other`
    unmapped_codec_ids: Vec<String>,
    tags: TagList,
    /// Per-track tags from `Targets`-scoped `Tag` elements: the `TagTrackUID`
    /// each names and that element's tags. One entry per scoped `Tag` element,
    /// appended as they parse (never merged), so a consumer can post the newly
    /// parsed ones by count.
    track_tags: Vec<(u64, TagList)>,
    /// Metadata a `TrackEntry` declares itself (`Name` / `Language`, M788), keyed
    /// by **track number**, not the `TagTrackUID` `track_tags` uses.
    track_entry_tags: Vec<(u64, TagList)>,
    /// The Segment `Chapters` table of contents (M1046), empty until it is seen.
    chapters: Vec<Chapter>,
    /// The current Timestamp of the open Cluster. `Some` while its children are
    /// being parsed at the top level, `None` otherwise.
    open_cluster_ts: Option<u64>,
    // set by damage inside a Cluster, cleared when the next Cluster ID turns up
    resyncing: bool,
    /// Absolute byte offset of `buf[0]` in the source stream (bytes consumed so
    /// far). Used once to fix `segment_data_pos`, the anchor for Cue positions.
    consumed: u64,
    /// Absolute byte offset of the Segment's data start (first byte after the
    /// Segment header), the anchor `CueClusterPosition` is relative to. `None`
    /// until the Segment header is seen; kept across a mid-segment seek.
    segment_data_pos: Option<u64>,
    /// The parsed `Cues` index (empty until the `Cues` element is seen), the
    /// time -> Cluster-position map [`cue_seek_offset`](Self::cue_seek_offset) uses.
    cues: Vec<CuePoint>,
    /// `Cues` byte position from a `SeekHead` (relative to the Segment data start),
    /// so the demuxer can prefetch an end-of-file `Cues` before reaching it
    /// (M374). `None` until a `SeekHead` indexing `Cues` is parsed.
    cues_index_pos: Option<u64>,
    completed: Vec<MkvFrame>,
}

impl Default for MatroskaDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl MatroskaDemuxer {
    pub fn new() -> Self {
        Self {
            buf: FifoBuffer::default(),
            in_segment: false,
            timestamp_scale: DEFAULT_TIMESTAMP_SCALE,
            duration_ns: None,
            tracks: Vec::new(),
            codec_privates: Vec::new(),
            block_encodings: Vec::new(),
            #[cfg(feature = "matroska-decrypt")]
            content_keys: Vec::new(),
            unmapped_codec_ids: Vec::new(),
            tags: TagList::new(),
            track_tags: Vec::new(),
            track_entry_tags: Vec::new(),
            chapters: Vec::new(),
            open_cluster_ts: None,
            resyncing: false,
            consumed: 0,
            segment_data_pos: None,
            cues: Vec::new(),
            cues_index_pos: None,
            completed: Vec::new(),
        }
    }

    // a WebM Encryption track whose ContentEncKeyID has a key here decrypts, any other stays flagged
    #[cfg(feature = "matroska-decrypt")]
    pub fn with_content_keys(mut self, keys: &[([u8; 16], [u8; 16])]) -> Self {
        self.content_keys = keys.iter().map(|(id, key)| (*id, AesKey(*key))).collect();
        self
    }

    #[cfg(feature = "matroska-decrypt")]
    fn content_decryption(&self, key_id: &[u8]) -> Option<ContentDecoding> {
        self.content_keys
            .iter()
            .rev()
            .find(|(id, _)| id[..] == *key_id)
            .map(|(_, key)| ContentDecoding::Decrypt(*key))
    }

    #[cfg(not(feature = "matroska-decrypt"))]
    fn content_decryption(&self, _key_id: &[u8]) -> Option<ContentDecoding> {
        None
    }

    // a fresh parser for a re-scan from offset 0 that keeps the content keys
    pub fn reset(&mut self) {
        #[cfg(feature = "matroska-decrypt")]
        let content_keys = core::mem::take(&mut self.content_keys);
        *self = Self::new();
        #[cfg(feature = "matroska-decrypt")]
        {
            self.content_keys = content_keys;
        }
    }

    /// The absolute byte offset of the `Cues` element located by a `SeekHead`
    /// (Segment data start + the indexed position), or `None` if no `SeekHead`
    /// pointing at `Cues` has been parsed. The demuxer byte-seeks here to prefetch
    /// an end-of-file `Cues` index before a seek (M374).
    pub fn cue_index_offset(&self) -> Option<u64> {
        Some(self.segment_data_pos?.saturating_add(self.cues_index_pos?))
    }

    /// The parsed `Cues` index (empty until the `Cues` element is seen).
    pub fn cues(&self) -> &[CuePoint] {
        &self.cues
    }

    /// The absolute byte offset to seek to for `target_ns` using the `Cues` index:
    /// the Segment data start plus the position of the Cluster with the largest
    /// `CueTime` at or before the target (the keyframe at/before it; the demuxer's
    /// re-sync then advances to the first keyframe >= target). `None` if no `Cues`
    /// have been parsed or the Segment offset is unknown, so the caller falls back
    /// to a re-scan from the start. A target before the first cue clamps to it.
    pub fn cue_seek_offset(&self, target_ns: u64) -> Option<u64> {
        let base = self.segment_data_pos?;
        if self.cues.is_empty() {
            return None;
        }
        let cue = self
            .cues
            .iter()
            .filter(|c| c.time_ns <= target_ns)
            .max_by_key(|c| c.time_ns)
            .or_else(|| self.cues.iter().min_by_key(|c| c.time_ns))?;
        Some(base.saturating_add(cue.cluster_position))
    }

    /// Reset for a mid-segment (Cue-indexed) seek: drop the byte buffer and any
    /// open-Cluster state, but keep what the file already established and the
    /// landing point does not re-send, the `Tracks` / `TimestampScale` / `Tags` /
    /// `Cues` / Segment offset. (A full re-scan from offset 0 uses [`new`](Self::new)
    /// instead, since it re-reads the EBML header and Tracks.)
    pub fn reset_keeping_tracks(&mut self) {
        self.buf.clear();
        self.open_cluster_ts = None;
        self.resyncing = false;
        self.completed.clear();
    }

    /// Consume `n` bytes from the front of `buf`, advancing the absolute position.
    fn consume(&mut self, n: usize) {
        self.buf.discard(n);
        self.consumed = self.consumed.saturating_add(n as u64);
    }

    // damage inside a Cluster: skip to the next Cluster ID, as ffmpeg does
    fn begin_resync(&mut self) {
        self.open_cluster_ts = None;
        self.resyncing = true;
    }

    // drops bytes up to the next Cluster ID, false while none is buffered yet
    fn skip_to_cluster(&mut self) -> bool {
        let cluster_id = ID_CLUSTER.to_be_bytes();
        let found = self
            .buf
            .as_slice()
            .windows(cluster_id.len())
            .position(|window| window == cluster_id);
        let Some(at) = found else {
            // keep a Cluster ID split across two pushes
            let partial_id_len = cluster_id.len() - 1;
            self.consume(self.buf.len().saturating_sub(partial_id_len));
            return false;
        };
        self.consume(at);
        self.resyncing = false;
        true
    }

    /// The elementary streams announced by `Tracks` (empty until it is seen).
    pub fn tracks(&self) -> &[MkvTrack] {
        &self.tracks
    }

    /// The container's total duration in nanoseconds, from the Segment `Info`
    /// `Duration`. `None` until Info parses, or when the file declares none.
    pub fn duration_ns(&self) -> Option<u64> {
        self.duration_ns
    }

    /// The `CodecPrivate` decoder-init bytes for a track number, if the track
    /// carried a non-empty one (FLAC's native `fLaC` STREAMINFO header).
    pub fn codec_private(&self, track: u64) -> Option<&[u8]> {
        self.codec_privates
            .iter()
            .find(|(n, _)| *n == track)
            .map(|(_, p)| p.as_slice())
    }

    pub fn unmapped_codec_ids(&self) -> &[String] {
        &self.unmapped_codec_ids
    }

    /// The stream metadata from the Segment's `Tags` element and the `Info`
    /// `Title` (empty until either is seen). Accumulates across pushes.
    pub fn tags(&self) -> &TagList {
        &self.tags
    }

    /// The `Targets`-scoped tags: one entry per track-scoped `Tag` element, the
    /// `TagTrackUID` it names paired with its tags (M787). The UID maps to a
    /// track through [`MkvTrack::uid`]; a UID no track carries stays here
    /// unresolved. Accumulates across pushes, in parse order.
    pub fn track_tags(&self) -> &[(u64, TagList)] {
        &self.track_tags
    }

    /// The metadata each `TrackEntry` declares itself, `Name` as [`Tag::Title`]
    /// and `Language` / `LanguageBCP47` as [`Tag::Language`] (M788), keyed by
    /// track number. This is where ffmpeg puts a stream's title and language, so
    /// a consumer merges it with [`track_tags`](Self::track_tags) to get one view
    /// per track. Empty until `Tracks` is parsed; a track declaring neither has
    /// no entry.
    pub fn track_entry_tags(&self) -> &[(u64, TagList)] {
        &self.track_entry_tags
    }

    /// The Segment `Chapters` table of contents (M1046), empty until a
    /// `Chapters` element is parsed. Times are stream-time nanoseconds; hidden
    /// editions and atoms are already filtered out.
    pub fn chapters(&self) -> &[Chapter] {
        &self.chapters
    }

    /// Drain the frames demuxed so far.
    pub fn take_frames(&mut self) -> Vec<MkvFrame> {
        core::mem::take(&mut self.completed)
    }

    /// Feed container bytes. Complete top-level elements are parsed as they
    /// arrive; a partial trailing element waits for the next call.
    pub fn push_data(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
        self.drain_elements();
    }

    /// Consume whole top-level elements from the front of `buf`. The Segment and
    /// each Cluster are descended into (their children are read at this level);
    /// every other element is consumed once its definite-size body is fully
    /// buffered.
    fn drain_elements(&mut self) {
        loop {
            if self.resyncing && !self.skip_to_cluster() {
                return;
            }
            let Some((id, id_len)) = read_id(self.buf.as_slice(), 0) else {
                return;
            };
            let Some((size, size_len, unknown)) = read_size(self.buf.as_slice(), id_len) else {
                return;
            };
            let header = id_len + size_len;

            if id == ID_SEGMENT {
                // Descend: a Segment's children are parsed at this level, so its
                // own size (definite or unknown) is never needed.
                self.consume(header);
                // Anchor for Cue cluster positions: the Segment data start. Fixed
                // once (a mid-segment seek lands past the header, never re-here).
                if self.segment_data_pos.is_none() {
                    self.segment_data_pos = Some(self.consumed);
                }
                self.in_segment = true;
                continue;
            }

            // A Cluster is likewise descended into, whatever its size: its children
            // are parsed at this level until the next top-level element ends it.
            if id == ID_CLUSTER {
                self.consume(header);
                self.open_cluster_ts = Some(0);
                continue;
            }

            // Inside an open Cluster, its own children are decoded
            // here; any other element closes the Cluster and is handled normally.
            if self.open_cluster_ts.is_some() {
                match id {
                    ID_TIMESTAMP | ID_SIMPLE_BLOCK | ID_BLOCK_GROUP => {
                        // a Cluster child must carry a definite size
                        let Some(total) = header.checked_add(size as usize).filter(|_| !unknown)
                        else {
                            self.begin_resync();
                            continue;
                        };
                        // checked on the buffered prefix, so a damaged size cannot stall the wait below
                        let buffered = &self.buf.as_slice()[header..total.min(self.buf.len())];
                        let names_unknown_track = id != ID_TIMESTAMP
                            && !self.tracks.is_empty()
                            && block_track_number(id, buffered)
                                .is_some_and(|n| !self.tracks.iter().any(|t| t.number == n));
                        if names_unknown_track {
                            self.begin_resync();
                            continue;
                        }
                        if self.buf.len() < total {
                            return;
                        }
                        if id == ID_TIMESTAMP {
                            self.open_cluster_ts =
                                Some(read_uint(&self.buf.as_slice()[header..total]));
                        } else {
                            let ts = self.open_cluster_ts.unwrap_or(0);
                            let mut frames = parse_block_element(
                                id,
                                &self.buf.as_slice()[header..total],
                                ts,
                                self.timestamp_scale,
                                &self.tracks,
                            );
                            decode_encodings(&mut frames, &mut self.block_encodings);
                            self.completed.extend(frames);
                        }
                        self.consume(total);
                        continue;
                    }
                    // A real level-1 sibling ends the Cluster; handle it below.
                    id if is_level1_element(id) => self.open_cluster_ts = None,
                    // A benign Cluster child (Void, CRC-32, Position, PrevSize) or
                    // an element we do not decode: skip its bytes and keep the
                    // Cluster open so the following blocks still parse.
                    _ => {
                        // need a definite size to skip
                        let Some(total) = header.checked_add(size as usize).filter(|_| !unknown)
                        else {
                            self.begin_resync();
                            continue;
                        };
                        if self.buf.len() < total {
                            return;
                        }
                        self.consume(total);
                        continue;
                    }
                }
            }

            // Every other element is consumed whole; a definite size tells us where
            // it ends (an unknown size here is a container we do not descend).
            if unknown {
                return;
            }
            let Some(total) = header.checked_add(size as usize) else {
                return;
            };
            if self.buf.len() < total {
                return; // wait for the rest of this element
            }

            if self.in_segment {
                match id {
                    ID_INFO => {
                        if let Some(scale) =
                            parse_timestamp_scale(&self.buf.as_slice()[header..total])
                        {
                            self.timestamp_scale = scale;
                        }
                        // Duration is in TimestampScale ticks, so it is read
                        // after the scale it is expressed in.
                        self.duration_ns = parse_info_duration(
                            &self.buf.as_slice()[header..total],
                            self.timestamp_scale,
                        );
                        if let Some(title) = parse_info_title(&self.buf.as_slice()[header..total]) {
                            self.tags.push(Tag::Title(title));
                        }
                    }
                    ID_TRACKS => {
                        let content_key = |key_id: &[u8]| self.content_decryption(key_id);
                        for parsed in
                            parse_tracks(&self.buf.as_slice()[header..total], &content_key)
                        {
                            self.tracks.push(parsed.track);
                            if !parsed.codec_private.is_empty() {
                                self.codec_privates
                                    .push((parsed.track.number, parsed.codec_private));
                            }
                            if !parsed.tags.is_empty() {
                                self.track_entry_tags
                                    .push((parsed.track.number, parsed.tags));
                            }
                            if !parsed.block_encodings.is_empty() {
                                self.block_encodings.push(TrackBlockDecoding {
                                    track: parsed.track.number,
                                    chain: parsed.block_encodings,
                                    dropped_block_logged: false,
                                });
                            }
                            if let Some(codec_id) = parsed.unmapped_codec_id {
                                self.unmapped_codec_ids.push(codec_id);
                            }
                        }
                    }
                    ID_TAGS => {
                        let scopes = parse_tags(&self.buf.as_slice()[header..total]);
                        for tag in scopes.global {
                            self.tags.push(tag);
                        }
                        for (uid, tags) in scopes.per_track {
                            self.track_tags.push((uid, tags.into_iter().collect()));
                        }
                    }
                    ID_CHAPTERS => {
                        self.chapters
                            .extend(parse_chapters(&self.buf.as_slice()[header..total]));
                    }
                    // The Cues index: time -> Cluster byte position, for indexed
                    // seeking (`cue_seek_offset`). Often trails the Clusters.
                    ID_CUES => {
                        self.cues =
                            parse_cues(&self.buf.as_slice()[header..total], self.timestamp_scale);
                    }
                    // A SeekHead (at the Segment start) locates the Cues element,
                    // so a seek can prefetch an end-of-file Cues before reaching it.
                    ID_SEEK_HEAD => {
                        if let Some(pos) = parse_seekhead_cues(&self.buf.as_slice()[header..total])
                        {
                            self.cues_index_pos = Some(pos);
                        }
                    }
                    _ => {} // Attachments / Void, etc.
                }
            }
            // (elements before the Segment, e.g. the EBML header, are skipped.)
            self.consume(total);
        }
    }
}

/// Parse a `SeekHead` for the `Cues` byte position (relative to the Segment data
/// start), if it indexes one. Each `Seek` entry pairs a `SeekID` (the target
/// element's ID, kept whole) with a `SeekPosition`; the entry whose `SeekID` is
/// `Cues` gives the position. `None` if `Cues` is not indexed.
fn parse_seekhead_cues(body: &[u8]) -> Option<u64> {
    for (id, seek) in children(body) {
        if id != ID_SEEK {
            continue;
        }
        let mut seek_id = None;
        let mut pos = None;
        for (sid, s) in children(seek) {
            match sid {
                ID_SEEK_ID => seek_id = read_id(s, 0).map(|(id, _)| id),
                ID_SEEK_POSITION => pos = Some(read_uint(s)),
                _ => {}
            }
        }
        if seek_id == Some(ID_CUES) {
            return pos;
        }
    }
    None
}

/// Parse a `Cues` element into its [`CuePoint`]s. Each `CuePoint` carries a
/// `CueTime` (scaled to ns here) and, in its `CueTrackPositions`, a
/// `CueClusterPosition` (kept relative to the Segment data start). Points missing
/// either field are skipped; the track of the position is ignored (v1: one
/// indexed track, the muxer writes a single set).
fn parse_cues(body: &[u8], timestamp_scale: u64) -> Vec<CuePoint> {
    let mut out = Vec::new();
    for (id, cue_point) in children(body) {
        if id != ID_CUE_POINT {
            continue;
        }
        let mut time = None;
        let mut pos = None;
        for (cid, c) in children(cue_point) {
            match cid {
                ID_CUE_TIME => time = Some(read_uint(c)),
                ID_CUE_TRACK_POSITIONS => {
                    for (pid, p) in children(c) {
                        if pid == ID_CUE_CLUSTER_POSITION {
                            pos = Some(read_uint(p));
                        }
                    }
                }
                _ => {}
            }
        }
        if let (Some(t), Some(p)) = (time, pos) {
            out.push(CuePoint {
                time_ns: t.saturating_mul(timestamp_scale),
                cluster_position: p,
            });
        }
    }
    out
}

/// Iterate the direct child elements of a master element body, yielding
/// `(id, contents)`. Stops at the first malformed or truncated child.
struct EbmlChildren<'a> {
    body: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for EbmlChildren<'a> {
    type Item = (u32, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        let (id, id_len) = read_id(self.body, self.pos)?;
        let (size, size_len, _unknown) = read_size(self.body, self.pos + id_len)?;
        let start = self.pos + id_len + size_len;
        let end = start.checked_add(size as usize)?;
        if end > self.body.len() {
            return None;
        }
        self.pos = end;
        Some((id, &self.body[start..end]))
    }
}

fn children(body: &[u8]) -> EbmlChildren<'_> {
    EbmlChildren { body, pos: 0 }
}

fn parse_timestamp_scale(info: &[u8]) -> Option<u64> {
    children(info)
        .find(|(id, _)| *id == ID_TIMESTAMP_SCALE)
        .map(|(_, d)| read_uint(d))
}

/// The Segment `Info` `Duration` in nanoseconds. The element is a float in
/// `TimestampScale` ticks; a negative, non-finite or absurdly large value is
/// refused rather than saturating into a nonsense length.
fn parse_info_duration(info: &[u8], timestamp_scale: u64) -> Option<u64> {
    let (_, data) = children(info).find(|(id, _)| *id == ID_DURATION)?;
    let ticks = read_float(data);
    if !ticks.is_finite() || ticks <= 0.0 || ticks > MAX_DURATION_TICKS {
        return None;
    }
    Some((ticks as u64).saturating_mul(timestamp_scale))
}

/// Largest `Duration` tick count accepted: at the default 1 ms scale this is a
/// little over a century, and it keeps the float-to-integer cast in range.
const MAX_DURATION_TICKS: f64 = 4.0e12;

/// The Segment `Info` `Title` (the whole-file title), if present and valid UTF-8.
fn parse_info_title(info: &[u8]) -> Option<String> {
    let (_, data) = children(info).find(|(id, _)| *id == ID_TITLE)?;
    core::str::from_utf8(data).ok().map(String::from)
}

/// How deep nested `SimpleTag`s are followed. The nesting comes from the stream,
/// so the walk is bounded instead of recursing to whatever depth a file claims.
const MAX_SIMPLE_TAG_DEPTH: u32 = 4;

/// A parsed Segment `Tags` element, split by what each `Tag`'s `Targets` scoped
/// it to.
#[derive(Debug, Default)]
struct MkvTagScopes {
    /// Tags of the whole stream: a `Tag` with no `Targets`, an empty one, or one
    /// naming `TagTrackUID` 0 (the spec's "all tracks").
    global: Vec<Tag>,
    /// One entry per track-scoped `Tag` element: the `TagTrackUID` it targets and
    /// that element's tags. A `Tag` targeting several tracks yields one entry per
    /// UID; two elements targeting one track stay two entries.
    per_track: Vec<(u64, Vec<Tag>)>,
}

/// Parse the Segment `Tags` element. Each `Tag`'s `SimpleTag` children carry a
/// `TagName` / `TagString` pair; the conventional uppercase Matroska names
/// (`TITLE`, `ARTIST`, ...) map through [`Tag::from_key_value`]. A nested
/// `SimpleTag` flattens to a `parent/child` key. The `Targets` child scopes the
/// whole `Tag`: a `TagTrackUID` sends its tags to that track, anything else
/// (absent, empty, UID 0) keeps them whole-stream.
fn parse_tags(body: &[u8]) -> MkvTagScopes {
    let mut out = MkvTagScopes::default();
    for (tag_id, tag) in children(body) {
        if tag_id != ID_TAG {
            continue;
        }
        let mut tags = Vec::new();
        for (sid, simple) in children(tag) {
            if sid == ID_SIMPLE_TAG {
                collect_simple_tag(simple, "", 0, &mut tags);
            }
        }
        if tags.is_empty() {
            continue;
        }
        let uids = target_track_uids(tag);
        if uids.is_empty() {
            out.global.extend(tags);
        } else {
            for uid in uids {
                out.per_track.push((uid, tags.clone()));
            }
        }
    }
    out
}

/// Cap on the tracks one `Tag` may target. The count comes from the stream and
/// each target duplicates the `Tag`'s tags, so it is bounded rather than trusted.
const MAX_TAG_TARGETS: usize = 64;

/// The `TagTrackUID`s a `Tag`'s `Targets` names, ignoring UID 0 (the spec's "all
/// tracks", i.e. no scoping). Empty when the `Tag` is not track-scoped: no
/// `Targets`, none naming a track, or only the all-tracks UID.
fn target_track_uids(tag: &[u8]) -> Vec<u64> {
    let mut uids = Vec::new();
    for (id, targets) in children(tag) {
        if id != ID_TARGETS {
            continue;
        }
        for (tid, data) in children(targets) {
            if tid == ID_TAG_TRACK_UID {
                let uid = read_uint(data);
                if uid != 0 && !uids.contains(&uid) {
                    uids.push(uid);
                    if uids.len() == MAX_TAG_TARGETS {
                        return uids;
                    }
                }
            }
        }
    }
    uids
}

/// Flatten one `SimpleTag` (and its nested `SimpleTag` children) into `out`. A
/// `TagName` keyed `TagString` value (both UTF-8) becomes one [`Tag`]; a nested
/// child's key is `parent/child`, so `ORIGINAL/TITLE` keeps both levels. A
/// `SimpleTag` with no name scopes nothing and is skipped whole; one with a name
/// but no (UTF-8) string still scopes its children, e.g. a `TagBinary` value.
fn collect_simple_tag(body: &[u8], prefix: &str, depth: u32, out: &mut Vec<Tag>) {
    let mut name: Option<&str> = None;
    let mut value: Option<&str> = None;
    for (id, data) in children(body) {
        match id {
            ID_TAG_NAME => name = core::str::from_utf8(data).ok(),
            ID_TAG_STRING => value = core::str::from_utf8(data).ok(),
            _ => {}
        }
    }
    let Some(name) = name else {
        return;
    };
    let key = if prefix.is_empty() {
        String::from(name)
    } else {
        alloc::format!("{prefix}/{name}")
    };
    if let Some(value) = value {
        out.push(Tag::from_key_value(&key, value));
    }
    if depth >= MAX_SIMPLE_TAG_DEPTH {
        return;
    }
    for (id, nested) in children(body) {
        if id == ID_SIMPLE_TAG {
            collect_simple_tag(nested, &key, depth + 1, out);
        }
    }
}

/// How deep nested `ChapterAtom`s are followed. The nesting comes from the
/// stream, so the walk is bounded instead of recursing to whatever depth a file
/// claims.
const MAX_CHAPTER_DEPTH: u32 = 4;

/// Cap on the chapters one `Chapters` element yields, nested ones included. An
/// atom costs the file a couple of bytes and costs the reader a whole
/// [`Chapter`], so the walk stops rather than turning a small crafted element
/// into a huge tree.
const MAX_CHAPTERS: usize = 4096;

/// Parse the Segment `Chapters` element. Each `EditionEntry` holds
/// `ChapterAtom`s, which may nest further atoms; every edition's atoms land in
/// one list, in file order. `ChapterTimeStart` / `ChapterTimeEnd` are already
/// nanoseconds (unlike Cluster timestamps, they are not scaled by
/// `TimestampScale`), and the first `ChapterDisplay` gives the title and its
/// `ChapLanguage`. A hidden edition or atom is skipped: it is not meant to reach
/// a chapter menu.
fn parse_chapters(body: &[u8]) -> Vec<Chapter> {
    let mut out = Vec::new();
    let mut budget = MAX_CHAPTERS;
    for (id, edition) in children(body) {
        if id != ID_EDITION_ENTRY || flag_is_set(edition, ID_EDITION_FLAG_HIDDEN) {
            continue;
        }
        collect_chapter_atoms(edition, 0, &mut budget, &mut out);
    }
    out
}

/// Whether a master element declares `flag` as a non-zero child.
fn flag_is_set(body: &[u8], flag: u32) -> bool {
    children(body).any(|(id, data)| id == flag && read_uint(data) != 0)
}

/// Append the `ChapterAtom` children of `body` to `out`, descending into nested
/// atoms until `depth` hits [`MAX_CHAPTER_DEPTH`] or `budget` runs out.
fn collect_chapter_atoms(body: &[u8], depth: u32, budget: &mut usize, out: &mut Vec<Chapter>) {
    for (id, atom) in children(body) {
        if id != ID_CHAPTER_ATOM || flag_is_set(atom, ID_CHAPTER_FLAG_HIDDEN) {
            continue;
        }
        if *budget == 0 {
            return;
        }
        *budget -= 1;
        let mut chapter = Chapter::default();
        let mut display_seen = false;
        for (child_id, data) in children(atom) {
            match child_id {
                ID_CHAPTER_TIME_START => chapter.start_ns = read_uint(data),
                ID_CHAPTER_TIME_END => chapter.end_ns = Some(read_uint(data)),
                ID_CHAPTER_DISPLAY if !display_seen => {
                    display_seen = true;
                    for (display_id, text) in children(data) {
                        match display_id {
                            ID_CHAP_STRING => {
                                chapter.title = bounded_string(text).unwrap_or("").into()
                            }
                            ID_CHAP_LANGUAGE => {
                                chapter.language = bounded_string(text).map(String::from)
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        if depth < MAX_CHAPTER_DEPTH {
            collect_chapter_atoms(atom, depth + 1, budget, &mut chapter.sub_chapters);
        }
        out.push(chapter);
    }
}

/// One parsed `TrackEntry`: the track, its `CodecPrivate` decoder-init bytes
/// (empty for codecs that carry none), and the metadata the entry itself
/// declares (`Name` / `Language`, M788).
#[derive(Debug)]
struct ParsedTrack {
    track: MkvTrack,
    codec_private: Vec<u8>,
    tags: TagList,
    /// The track's block `ContentEncoding` chain, kept beside `track` so
    /// `MkvTrack` stays `Copy` (the header-stripping settings are owned bytes).
    block_encodings: Vec<ContentDecoding>,
    unmapped_codec_id: Option<String>,
}

fn parse_tracks(body: &[u8], content_key: ContentKeyLookup<'_>) -> Vec<ParsedTrack> {
    let mut tracks = Vec::new();
    for (id, entry) in children(body) {
        if id == ID_TRACK_ENTRY {
            if let Some(t) = parse_track_entry(entry, content_key) {
                tracks.push(t);
            }
        }
    }
    tracks
}

/// Cap on a short string element surfaced to the application (a `TrackEntry`
/// `Name` / `Language`, a `ChapString`). The length is the file's claim, so an
/// absurd one is ignored rather than copied.
const MAX_STRING_ELEMENT_LEN: usize = 4096;

/// A short string element's text: UTF-8 and within the length cap, else nothing.
fn bounded_string(data: &[u8]) -> Option<&str> {
    if data.len() > MAX_STRING_ELEMENT_LEN {
        return None;
    }
    core::str::from_utf8(data).ok()
}

fn parse_track_entry(body: &[u8], content_key: ContentKeyLookup<'_>) -> Option<ParsedTrack> {
    let mut number = 0u64;
    let mut uid = 0u64;
    let mut name: Option<&str> = None;
    let mut language: Option<&str> = None;
    let mut language_bcp47: Option<&str> = None;
    let mut codec_id: &[u8] = &[];
    let mut codec_private: &[u8] = &[];
    let mut width = 0u32;
    let mut height = 0u32;
    let mut colour_space: Option<[u8; 4]> = None;
    let mut channels = 0u8;
    let mut sample_rate = 0u32;
    let mut bit_depth = 0u64;
    let mut default_duration_ns = 0u64;
    // no ContentEncodings element means nothing to undo
    let mut encodings = Some(ContentEncodingChains::default());
    for (id, data) in children(body) {
        match id {
            ID_TRACK_NUMBER => number = read_uint(data),
            // The id a Tags `Targets` scopes a per-track tag to.
            ID_TRACK_UID => uid = read_uint(data),
            // The track's own metadata: where ffmpeg puts a stream's title and
            // language (a `Tags` element carries the rest).
            ID_TRACK_NAME => name = bounded_string(data),
            ID_LANGUAGE => language = bounded_string(data),
            ID_LANGUAGE_BCP47 => language_bcp47 = bounded_string(data),
            ID_CODEC_ID => codec_id = data,
            // decoder-init bytes (FLAC's fLaC STREAMINFO); kept per track number.
            ID_CODEC_PRIVATE => codec_private = data,
            ID_DEFAULT_DURATION => default_duration_ns = read_uint(data),
            ID_VIDEO => {
                for (vid, vdata) in children(data) {
                    match vid {
                        ID_PIXEL_WIDTH => width = read_uint(vdata) as u32,
                        ID_PIXEL_HEIGHT => height = read_uint(vdata) as u32,
                        ID_COLOUR_SPACE => colour_space = vdata.try_into().ok(),
                        _ => {}
                    }
                }
            }
            ID_AUDIO => {
                for (aid, adata) in children(data) {
                    match aid {
                        ID_CHANNELS => channels = read_uint(adata) as u8,
                        ID_SAMPLING_FREQ => sample_rate = read_float(adata) as u32,
                        ID_BIT_DEPTH => bit_depth = read_uint(adata),
                        _ => {}
                    }
                }
            }
            ID_CONTENT_ENCODINGS => encodings = parse_content_encodings(data, content_key),
            _ => {} // TrackType is implied by the CodecID prefix; FlagLacing etc. ignored
        }
    }
    if number == 0 {
        return None;
    }
    let (unsupported_encoding, chains) = match encodings {
        Some(chains) => (false, chains),
        None => (true, ContentEncodingChains::default()),
    };
    // ffmpeg too drops a CodecPrivate that does not decode rather than pass on the stored bytes
    let codec_private = if codec_private.is_empty() || chains.codec_private.is_empty() {
        codec_private.to_vec()
    } else {
        undo_encodings(
            &chains.codec_private,
            codec_private.to_vec(),
            MAX_INFLATED_CODEC_PRIVATE_LEN,
        )
        .unwrap_or_default()
    };
    let mut tags = TagList::new();
    if let Some(name) = name {
        tags.push(Tag::Title(String::from(name)));
    }
    // Only an element that is actually there becomes a tag: the spec's implicit
    // "eng" default for a missing Language is not metadata the file stated.
    if let Some(lang) = language_bcp47.or(language) {
        tags.push(Tag::Language(String::from(lang)));
    }
    let codec = match MkvCodec::from_codec_id(codec_id) {
        MkvCodec::Other => pcm_codec(codec_id, bit_depth)
            .or_else(|| vfw_codec(codec_id, &codec_private))
            .or_else(|| uncompressed_codec(codec_id, colour_space?))
            .or_else(|| prores_codec(codec_id, &codec_private))
            .or_else(|| tagged_codec(codec_id, &codec_private))
            .unwrap_or(MkvCodec::Other),
        codec => codec,
    };
    Some(ParsedTrack {
        unmapped_codec_id: unmapped_av_codec_id(codec, codec_id, &codec_private),
        track: MkvTrack {
            number,
            uid,
            codec,
            width,
            height,
            channels,
            sample_rate,
            default_duration_ns,
            unsupported_encoding,
        },
        codec_private,
        tags,
        block_encodings: chains.block,
    })
}

// the sample format of an `A_PCM/*` track, `None` for another CodecID or a depth no PCM format has
fn pcm_codec(codec_id: &[u8], bit_depth: u64) -> Option<MkvCodec> {
    use crate::waveformat::{audio_format, WAVE_FORMAT_IEEE_FLOAT, WAVE_FORMAT_PCM};
    let (wave_format_tag, big_endian) = match without_nul_padding(codec_id) {
        b"A_PCM/INT/LIT" => (WAVE_FORMAT_PCM, false),
        b"A_PCM/INT/BIG" => (WAVE_FORMAT_PCM, true),
        b"A_PCM/FLOAT/IEEE" => (WAVE_FORMAT_IEEE_FLOAT, false),
        _ => return None,
    };
    let format = audio_format(wave_format_tag, u16::try_from(bit_depth).ok()?)?;
    Some(MkvCodec::Pcm { format, big_endian })
}

const VFW_CODEC_ID: &str = "V_MS/VFW/FOURCC";
const BITMAPINFO_COMPRESSION: core::ops::Range<usize> = 16..20;

fn without_nul_padding(codec_id: &[u8]) -> &[u8] {
    let padding = codec_id.iter().rev().take_while(|&&byte| byte == 0).count();
    &codec_id[..codec_id.len() - padding]
}

// the fourcc a `V_MS/VFW/FOURCC` track's BITMAPINFOHEADER names
fn vfw_fourcc<'a>(codec_id: &[u8], codec_private: &'a [u8]) -> Option<&'a [u8]> {
    (without_nul_padding(codec_id) == VFW_CODEC_ID.as_bytes())
        .then(|| codec_private.get(BITMAPINFO_COMPRESSION))
        .flatten()
}

fn vfw_codec(codec_id: &[u8], codec_private: &[u8]) -> Option<MkvCodec> {
    let fourcc = vfw_fourcc(codec_id, codec_private)?;
    if fourcc.eq_ignore_ascii_case(b"MJPG") {
        return Some(MkvCodec::Mjpeg);
    }
    crate::uncompressedvideo::raw_video_format(fourcc.try_into().ok()?).map(MkvCodec::RawVideo)
}

fn prores_codec(codec_id: &[u8], codec_private: &[u8]) -> Option<MkvCodec> {
    if without_nul_padding(codec_id) != b"V_PRORES" {
        return None;
    }
    // without a fourcc ffmpeg and GStreamer both decode the track as standard 4:2:2
    if codec_private.is_empty() {
        return Some(MkvCodec::ProRes(ProResVariant::Standard));
    }
    ProResVariant::from_fourcc(codec_private.try_into().ok()?).map(MkvCodec::ProRes)
}

fn uncompressed_codec(codec_id: &[u8], colour_space: [u8; 4]) -> Option<MkvCodec> {
    (without_nul_padding(codec_id) == b"V_UNCOMPRESSED")
        .then(|| crate::uncompressedvideo::raw_video_format(colour_space))
        .flatten()
        .map(MkvCodec::RawVideo)
}

// the wrappers whose `CodecPrivate` carries an AVI or QuickTime codec header, which a tagged codec rides
#[cfg(feature = "ffmpeg-long-tail")]
fn tagged_codec(codec_id: &[u8], codec_private: &[u8]) -> Option<MkvCodec> {
    const BITMAPINFO_LEN: usize = 40;
    const WAVEFORMAT_LEN: usize = 14;
    const WAVEFORMAT_TAG: core::ops::Range<usize> = 0..2;
    const SAMPLE_ENTRY_FOURCC: core::ops::Range<usize> = 4..8;
    const VISUAL_SAMPLE_ENTRY_LEN: usize = 86;
    let tag = |namespace, bytes: &[u8]| {
        let mut word = [0u8; 4];
        word[..bytes.len()].copy_from_slice(bytes);
        CodecTag {
            namespace,
            tag: u32::from_le_bytes(word),
        }
    };
    match without_nul_padding(codec_id) {
        id if id == VFW_CODEC_ID.as_bytes() && codec_private.len() >= BITMAPINFO_LEN => {
            Some(MkvCodec::TaggedVideo(tag(
                TagNamespace::Riff,
                &codec_private[BITMAPINFO_COMPRESSION],
            )))
        }
        b"A_MS/ACM" if codec_private.len() >= WAVEFORMAT_LEN => Some(MkvCodec::TaggedAudio(tag(
            TagNamespace::Riff,
            &codec_private[WAVEFORMAT_TAG],
        ))),
        b"V_QUICKTIME" if codec_private.len() >= VISUAL_SAMPLE_ENTRY_LEN => Some(
            MkvCodec::TaggedVideo(tag(TagNamespace::Mov, &codec_private[SAMPLE_ENTRY_FOURCC])),
        ),
        _ => None,
    }
}

#[cfg(not(feature = "ffmpeg-long-tail"))]
fn tagged_codec(_codec_id: &[u8], _codec_private: &[u8]) -> Option<MkvCodec> {
    None
}

// the packet count less one, the 255-run sizes of all but the last packet, then the packets
pub(crate) fn split_xiph_codec_private(private: &[u8]) -> Option<Vec<&[u8]>> {
    let (&count_less_one, laced) = private.split_first()?;
    split_xiph(laced, usize::from(count_less_one) + 1)
}

fn unmapped_av_codec_id(codec: MkvCodec, codec_id: &[u8], codec_private: &[u8]) -> Option<String> {
    if codec != MkvCodec::Other {
        return None;
    }
    let name = bounded_string(without_nul_padding(codec_id))?;
    let audio_or_video =
        name.starts_with(AUDIO_CODEC_ID_PREFIX) || name.starts_with(VIDEO_CODEC_ID_PREFIX);
    if !audio_or_video {
        return None;
    }
    let fourcc = vfw_fourcc(codec_id, codec_private)
        .filter(|fourcc| fourcc.iter().all(u8::is_ascii_graphic))
        .and_then(|fourcc| core::str::from_utf8(fourcc).ok());
    Some(match fourcc {
        Some(fourcc) => alloc::format!("{name} {fourcc}"),
        None => String::from(name),
    })
}

const AUDIO_CODEC_ID_PREFIX: &str = "A_";
const VIDEO_CODEC_ID_PREFIX: &str = "V_";

/// Parse a `BlockGroup`'s frames, carrying its `BlockDuration` (the cue display
/// window, essential for a subtitle track) onto them. Scans the children for the
/// `Block`, the optional `BlockDuration` and the optional `DiscardPadding` (any
/// order), then de-frames the block with the resulting duration.
fn parse_block_group(
    group: &[u8],
    cluster_ts: u64,
    scale: u64,
    tracks: &[MkvTrack],
    out: &mut Vec<MkvFrame>,
) {
    let mut block: Option<&[u8]> = None;
    let mut duration_raw = 0u64;
    let mut discard_ns = 0i64;
    for (bid, bdata) in children(group) {
        match bid {
            ID_BLOCK => block = Some(bdata),
            ID_BLOCK_DURATION => duration_raw = read_uint(bdata),
            ID_DISCARD_PADDING => discard_ns = read_int(bdata),
            _ => {}
        }
    }
    let Some(b) = block else {
        return;
    };
    // BlockDuration is in TimestampScale ticks, like the block timestamp.
    let duration_ns = duration_raw.saturating_mul(scale);
    let first = out.len();
    parse_block(b, cluster_ts, scale, tracks, duration_ns, out);
    apply_discard_padding(&mut out[first..], discard_ns);
}

/// Apply a `DiscardPadding` to the block's last frame: the ns of decoded audio
/// to drop from its tail, which is how Matroska spells the end-of-stream trim an
/// Ogg granule carries (the whole point of the element, RFC 7845 §4.4 in
/// Matroska's binding). Unlike `BlockDuration` it is nanoseconds, so it survives
/// the millisecond `TimestampScale` grid exactly, and it wins where both are
/// present, as in every ffmpeg-written file.
///
/// Only Opus is converted: the packet's own length has to be known to turn a
/// tail discard into the kept duration, and the Opus TOC byte is the only one
/// this parser can read. Everything about the value is the file's claim, so a
/// negative discard (the spec's leading-padding form, which a kept-samples
/// count cannot express), one no shorter than the packet, or a packet whose
/// length does not parse leaves the frame's duration untouched.
fn apply_discard_padding(frames: &mut [MkvFrame], discard_ns: i64) {
    if discard_ns <= 0 {
        return;
    }
    let Some(frame) = frames.last_mut() else {
        return;
    };
    if frame.codec != MkvCodec::Opus {
        return;
    }
    let samples = u64::from(crate::opusparse::packet_samples(&frame.data));
    let packet_ns = samples * 1_000_000_000 / u64::from(crate::opusparse::OPUS_RATE_HZ);
    if packet_ns == 0 || packet_ns <= discard_ns as u64 {
        return;
    }
    frame.duration_ns = packet_ns - discard_ns as u64;
}

// the track number a SimpleBlock or a BlockGroup's Block names, `None` when it does not parse
fn block_track_number(id: u32, body: &[u8]) -> Option<u64> {
    let block = if id == ID_BLOCK_GROUP {
        children(body).find(|(child, _)| *child == ID_BLOCK)?.1
    } else {
        body
    };
    read_size(block, 0).map(|(track, _, _)| track)
}

/// Parse a single Cluster child block element (a `SimpleBlock` or `BlockGroup`)
/// into frames, as the Cluster's children are decoded one at a time. The Cluster
/// `Timestamp` is handled by the caller.
fn parse_block_element(
    id: u32,
    body: &[u8],
    cluster_ts: u64,
    scale: u64,
    tracks: &[MkvTrack],
) -> Vec<MkvFrame> {
    let mut frames = Vec::new();
    match id {
        ID_SIMPLE_BLOCK => parse_block(body, cluster_ts, scale, tracks, 0, &mut frames),
        ID_BLOCK_GROUP => parse_block_group(body, cluster_ts, scale, tracks, &mut frames),
        _ => {}
    }
    frames
}

/// Parse a (Simple)Block, appending its frame(s): a track-number VINT, a 2-byte
/// signed relative timestamp, a flags byte, then the frame data. A laced block
/// carries several frames (Xiph / EBML / fixed lacing); they are spaced by the
/// track's `DefaultDuration` from the block timestamp when it is known, else they
/// share the block timestamp. A malformed block is dropped.
fn parse_block(
    block: &[u8],
    cluster_ts: u64,
    scale: u64,
    tracks: &[MkvTrack],
    duration_ns: u64,
    out: &mut Vec<MkvFrame>,
) {
    let Some((track, tn_len, _)) = read_size(block, 0) else {
        return;
    };
    let mut pos = tn_len;
    if pos + 3 > block.len() {
        return;
    }
    let rel = i16::from_be_bytes([block[pos], block[pos + 1]]);
    pos += 2;
    let flags = block[pos];
    pos += 1;
    let Some(t) = tracks.iter().find(|t| t.number == track) else {
        return;
    };
    let codec = t.codec;
    let default_duration_ns = t.default_duration_ns;
    // `cluster_ts` is an untrusted u64; cast-to-i64 can be negative and the add
    // can overflow, so saturate (the `abs < 0` clamp below still maps it to 0).
    let abs = (cluster_ts as i64).saturating_add(rel as i64);
    let pts_ns = if abs < 0 {
        0
    } else {
        (abs as u64).saturating_mul(scale)
    };
    let keyframe = flags & 0x80 != 0;

    let body = &block[pos..];
    let lacing = (flags >> 1) & 0x03;
    let frames = if lacing == 0 {
        alloc::vec![body]
    } else {
        match split_laced(body, lacing) {
            Some(v) => v,
            None => return, // malformed lacing: drop the block
        }
    };
    // A single (unlaced) frame keeps the block timestamp; laced frames advance by
    // DefaultDuration when known (i == 0 leaves the first at the block timestamp).
    for (i, data) in frames.into_iter().enumerate() {
        // default_duration_ns is untrusted; saturate the spacing multiply too,
        // not just the add.
        let frame_pts = pts_ns.saturating_add((i as u64).saturating_mul(default_duration_ns));
        out.push(MkvFrame {
            track,
            codec,
            pts_ns: frame_pts,
            duration_ns,
            keyframe,
            data: data.to_vec(),
        });
    }
}

/// Split a laced block body (`[frame_count-1][size headers][frame data]`) into
/// per-frame slices. `lacing` is the 2-bit field: 1 = Xiph, 2 = fixed, 3 = EBML.
fn split_laced(body: &[u8], lacing: u8) -> Option<Vec<&[u8]>> {
    let (&count_minus_1, rest) = body.split_first()?;
    let count = count_minus_1 as usize + 1;
    match lacing {
        1 => split_xiph(rest, count),
        2 => split_fixed(rest, count),
        3 => split_ebml(rest, count),
        _ => None,
    }
}

/// Fixed-size lacing: every frame is `len / count` bytes (exact division).
fn split_fixed(data: &[u8], count: usize) -> Option<Vec<&[u8]>> {
    if count == 0 || data.is_empty() || !data.len().is_multiple_of(count) {
        return None;
    }
    Some(data.chunks(data.len() / count).collect())
}

/// Xiph lacing: the first `count - 1` frame sizes are coded as 255-continuation
/// byte runs; the last frame is the remainder.
fn split_xiph(data: &[u8], count: usize) -> Option<Vec<&[u8]>> {
    let mut sizes = Vec::with_capacity(count);
    let mut pos = 0;
    for _ in 0..count - 1 {
        let mut size = 0usize;
        loop {
            let b = *data.get(pos)?;
            pos += 1;
            size += b as usize;
            if b != 0xFF {
                break;
            }
        }
        sizes.push(size);
    }
    slice_frames(data, pos, &sizes)
}

/// EBML lacing: the first frame size is an unsigned VINT, each subsequent size a
/// signed VINT delta from the previous; the last frame is the remainder. A signed
/// VINT of byte-length `n` decodes as `unsigned - (2^(7n-1) - 1)`.
fn split_ebml(data: &[u8], count: usize) -> Option<Vec<&[u8]>> {
    let coded = count.checked_sub(1)?;
    let mut sizes = Vec::with_capacity(count);
    let mut pos = 0;
    let mut cur = 0i64;
    for i in 0..coded {
        let (raw, len, _) = read_size(data, pos)?;
        pos += len;
        if i == 0 {
            cur = raw as i64;
        } else {
            let bias = (1i64 << (7 * len - 1)) - 1;
            // Untrusted deltas accumulated over up to 255 laces can overflow i64;
            // a malformed lacing fails the parse rather than panicking in debug.
            cur = cur.checked_add(raw as i64 - bias)?;
        }
        if cur < 0 {
            return None;
        }
        sizes.push(cur as usize);
    }
    slice_frames(data, pos, &sizes)
}

/// Slice frames out of `data` starting at `start`: one per entry in `sizes`, then
/// a final frame holding the remainder (so `sizes.len() + 1` frames total).
fn slice_frames<'a>(data: &'a [u8], start: usize, sizes: &[usize]) -> Option<Vec<&'a [u8]>> {
    let mut frames = Vec::with_capacity(sizes.len() + 1);
    let mut off = start;
    for &sz in sizes {
        let end = off.checked_add(sz)?;
        frames.push(data.get(off..end)?);
        off = end;
    }
    frames.push(data.get(off..)?);
    Some(frames)
}

/// Read an EBML element ID (1..4 bytes, length marker kept).
fn read_id(data: &[u8], pos: usize) -> Option<(u32, usize)> {
    let first = *data.get(pos)?;
    let len = match first {
        0x80..=0xFF => 1,
        0x40..=0x7F => 2,
        0x20..=0x3F => 3,
        0x10..=0x1F => 4,
        _ => return None,
    };
    if pos + len > data.len() {
        return None;
    }
    let mut id = 0u32;
    for &b in &data[pos..pos + len] {
        id = (id << 8) | b as u32;
    }
    Some((id, len))
}

/// Read an EBML variable-length size / integer (marker stripped). Returns
/// `(value, byte_len, is_unknown_size)`.
fn read_size(data: &[u8], pos: usize) -> Option<(u64, usize, bool)> {
    let first = *data.get(pos)?;
    if first == 0 {
        return None; // a leading zero byte would encode a length over 8 bytes
    }
    let len = first.leading_zeros() as usize + 1;
    if len > 8 || pos + len > data.len() {
        return None;
    }
    let value_mask = (1u64 << (8 - len)) - 1;
    let mut value = (first as u64) & value_mask;
    for &b in &data[pos + 1..pos + len] {
        value = (value << 8) | b as u64;
    }
    let unknown = value == (1u64 << (7 * len)) - 1;
    Some((value, len, unknown))
}

/// Read a big-endian unsigned integer element body (1..8 bytes).
fn read_uint(data: &[u8]) -> u64 {
    let mut v = 0u64;
    for &b in data.iter().take(8) {
        v = (v << 8) | b as u64;
    }
    v
}

/// Read a signed integer element body (big-endian two's complement, 1..8 bytes;
/// `0` for an empty or oversized one). The width is the file's, so the value is
/// sign-extended from whatever it gave.
fn read_int(data: &[u8]) -> i64 {
    if data.is_empty() || data.len() > 8 {
        return 0;
    }
    let mut v = if data[0] & 0x80 != 0 { -1i64 } else { 0 };
    for &b in data {
        v = (v << 8) | i64::from(b);
    }
    v
}

/// Read an IEEE-754 float element body (4 or 8 bytes; 0 otherwise).
fn read_float(data: &[u8]) -> f64 {
    match data.len() {
        4 => f32::from_be_bytes([data[0], data[1], data[2], data[3]]) as f64,
        8 => f64::from_be_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]),
        _ => 0.0,
    }
}

// --- Muxing (M115): the inverse of the demuxer above. ---

/// `CodecPrivate` element ID (the per-track decoder init bytes: avcC / hvcC
/// record for H.26x, the AudioSpecificConfig for AAC). Written by the muxer only;
/// the demuxer leaves these codecs' parameter sets in-band.
const ID_CODEC_PRIVATE: u32 = 0x63A2;

/// The track parameters a [`MatroskaMuxer`] writes. Geometry is used for video,
/// channels / sample_rate for audio (the codec selects which).
#[derive(Debug, Clone, Copy)]
pub struct MkvTrackSpec {
    pub codec: MkvCodec,
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub sample_rate: u32,
}

/// One track of a multi-track [`MatroskaMuxer`]: its parameters plus the
/// `CodecPrivate` decoder-init bytes (the avcC / hvcC record for H.26x, the
/// AudioSpecificConfig for AAC). Empty `codec_private` writes no element, which
/// suits codecs that need none (VP8 / VP9).
#[derive(Debug, Clone)]
pub struct MkvTrackConfig {
    pub spec: MkvTrackSpec,
    pub codec_private: Vec<u8>,
}

impl MkvTrackSpec {
    /// A subtitle track: the codec alone, since such a track has neither video
    /// geometry nor audio parameters.
    pub fn subtitle(codec: MkvCodec) -> Self {
        Self {
            codec,
            width: 0,
            height: 0,
            channels: 0,
            sample_rate: 0,
        }
    }
}

// --- Subtitle track mapping, shared by both muxer elements (M898 / M927 / M928).
// The single-track `crate::mkvmux::MkvMux` and the fan-in `crate::mkvmuxn::MkvMuxN`
// write subtitle pads identically, so the mapping lives here rather than in either.

/// The Matroska codec a bitmap subtitle format is written as, or `None` for one
/// these muxers do not write (a `S_HDMV/PGS` block needs the `.sup` framing
/// stripped, which no write path does yet).
pub fn subpicture_mkv_codec(format: SubPictureFormat) -> Option<MkvCodec> {
    match format {
        SubPictureFormat::VobSub => Some(MkvCodec::VobSub),
        SubPictureFormat::DvbSub => Some(MkvCodec::DvbSub),
        _ => None,
    }
}

/// The `CodecPrivate` a bitmap subtitle pad's in-band config blob carries: the
/// `.idx` text a VobSub track needs, normalized to the size / palette lines a
/// container holds (the cue index is a sidecar's file offset table), or the
/// five-byte page-id blob a DVB track needs. `None` when the bytes are a cue
/// rather than a config, which is what tells the two apart on one pad.
pub fn subpicture_config(format: SubPictureFormat, data: &[u8]) -> Option<Vec<u8>> {
    match format {
        SubPictureFormat::VobSub => Some(idx_config_text(&parse_idx(data)?).into_bytes()),
        SubPictureFormat::DvbSub => {
            let ids = parse_page_ids(data)?;
            let subtitling_type = data.get(4).copied().unwrap_or(DEFAULT_SUBTITLING_TYPE);
            Some(Vec::from(page_id_blob(ids, subtitling_type)))
        }
        _ => None,
    }
}

/// The block payload for one bitmap subtitle cue. An `S_DVBSUB` block is the
/// display set's bare segments: the PES data-field header a transport stream
/// wraps them in does not belong in a Matroska block, so a stream out of
/// `tsdemux` is unwrapped here. A VobSub block is the subpicture unit verbatim.
pub fn subpicture_block(format: SubPictureFormat, data: &[u8]) -> Vec<u8> {
    match format {
        SubPictureFormat::DvbSub => segment_span(data).to_vec(),
        _ => data.to_vec(),
    }
}

/// The `S_DVBSUB` `CodecPrivate` for a stream that names no pages of its own:
/// the `dvbsub-page-id` page as both the composition and the ancillary one.
pub fn default_page_blob(page_id: u16) -> Vec<u8> {
    Vec::from(page_id_blob(
        PageIds {
            composition: page_id,
            ancillary: page_id,
        },
        DEFAULT_SUBTITLING_TYPE,
    ))
}

/// The `CodecPrivate` a timed-text track carries in a storage syntax: the ASS
/// script header naming the fields each block holds, and nothing for plain text.
pub fn text_codec_private(format: TextFormat) -> Vec<u8> {
    match format {
        TextFormat::Ssa => Vec::from(ASS_SCRIPT_HEADER.as_bytes()),
        _ => Vec::new(),
    }
}

/// The `subtitle-format` property value naming a storage syntax, or `None` for a
/// format no `S_TEXT/*` mapping covers.
pub fn subtitle_format_str(format: TextFormat) -> Option<&'static str> {
    match format {
        TextFormat::Utf8 => Some("utf8"),
        TextFormat::Ssa => Some("ass"),
        _ => None,
    }
}

/// The storage syntax a `subtitle-format` property value names, the inverse of
/// [`subtitle_format_str`].
pub fn subtitle_format_from_str(value: &str) -> Option<TextFormat> {
    match value {
        "utf8" => Some(TextFormat::Utf8),
        "ass" | "ssa" => Some(TextFormat::Ssa),
        _ => None,
    }
}

/// Default cap on a Cluster's time span (ms): a new Cluster opens once a frame is
/// this far past the current Cluster's base timestamp.
const DEFAULT_MAX_CLUSTER_SPAN_MS: u64 = 1_000;

/// The reserved EBML size a two-pass element is written with and its finalize
/// fills in: eight bytes, which is also the all-ones "unknown size" a streaming
/// Segment keeps. Eight is the widest form, so any length fits without moving a
/// byte of what follows.
const UNKNOWN_SIZE_8: [u8; 8] = [0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];

/// A reserved size field waiting on its element's length: where the 8 bytes sit
/// in the muxed output, and where the element's data begins (the length is the
/// distance from there to the element's end).
#[derive(Debug, Clone, Copy)]
pub struct SizePatch {
    pub size_at: usize,
    pub data_at: usize,
}

/// Matroska / WebM multiplexer for a single track (M115): writes the EBML header,
/// the Segment, Info + Tracks, then time-windowed Clusters of frames. The inverse
/// of [`MatroskaDemuxer`]; the [`crate::mkvmux::MkvMux`] element wraps it.
///
/// Frames batch into one Cluster until one is more than the span cap past its
/// base timestamp (or time runs backward), amortizing the per-Cluster overhead.
///
/// Streaming (the default), the Segment and each Cluster carry the unknown-size
/// marker (the live shape, M143): a Cluster header opens the window and the next
/// one ends it, so frames go out incrementally with nothing held back to learn a
/// length and nothing patched afterwards. The muxer still records a `Cues` index
/// (one entry per Cluster holding a keyframe on the cue track) and emits it from
/// [`finish`](Self::finish) at EOS, so the stream is seekable once read to the
/// end (M375).
///
/// [`with_two_pass`](Self::with_two_pass) is for a caller buffering the whole
/// file: the Segment and every Cluster reserve an 8-byte size, and a front
/// `SeekHead` and the `Info` `Duration` are reserved too, all filled in by
/// [`finalize_seekable`]. The file then declares its bounds and seeks from byte 0
/// without reading past the Clusters (the M374 `SeekHead` prefetch), which is the
/// shape ffmpeg and GStreamer write.
///
/// Scope: default TimestampScale (1 ms). One or more tracks (see
/// [`MatroskaMuxer::new_multi`], the A/V case driven by
/// [`crate::mkvmuxn::MkvMuxN`]).
#[derive(Debug)]
pub struct MatroskaMuxer {
    /// One or more tracks; the Nth (0-based) writes Matroska TrackNumber N+1.
    tracks: Vec<MkvTrackConfig>,
    tags: TagList,
    /// Per-track tags (track index, tags), each written as a `Targets`-scoped
    /// `Tag` inside the same `Tags` element.
    track_tags: Vec<(usize, TagList)>,
    /// The table of contents, written as a `Chapters` element (M1046).
    chapters: Vec<Chapter>,
    max_cluster_span_ms: u64,
    header_written: bool,
    /// The open Cluster's base Timestamp (ms), or `None` before the first frame.
    cluster_base_ms: Option<u64>,
    /// The track (0-based) the `Cues` index references: the first video track, or
    /// track 0 if none. Cues conventionally index the video keyframes.
    cue_track: usize,
    /// Running byte offset into the Segment data (the byte after the Segment
    /// header, where `CueClusterPosition` is anchored): bytes emitted past it so
    /// far. Tracks the position of each Cluster as it streams out.
    segment_pos: u64,
    /// Byte offset (relative to the Segment data start) of the currently open
    /// Cluster, the `CueClusterPosition` a keyframe in it records.
    current_cluster_pos: u64,
    /// `Cues` entries collected so far: `(CueTime in TimestampScale units,
    /// CueClusterPosition relative to the Segment data start)`.
    cues: Vec<(u64, u64)>,
    /// The Cluster position of the last recorded cue, so at most one cue is kept
    /// per Cluster (the first keyframe in it), bounding the index size.
    last_cued_cluster_pos: Option<u64>,
    /// Whether to collect the `Cues` entries at all. A live caller never calls
    /// [`finish`](Self::finish), so recording them would grow without bound for
    /// as long as the stream runs; [`without_cues`](Self::without_cues) turns it
    /// off. Does not change a byte of the output either way.
    record_cues: bool,
    /// The two-pass (buffering) mode. It writes a front `SeekHead` (first
    /// element of the Segment data) indexing Info / Tracks / Tags / Cues,
    /// reserves the `Info` `Duration`, and gives the Segment and every Cluster a
    /// reserved 8-byte size field instead of the streaming unknown-size marker.
    /// All of them are placeholders the caller patches at EOS
    /// ([`finalize_seekable`]) with the whole output in hand, which the streaming
    /// path never has: the finished file then seeks from byte 0 without reading
    /// past the Clusters, declares its length, and bounds every element the way
    /// ffmpeg and GStreamer write one.
    two_pass: bool,
    /// Byte offset (in the muxed output, from byte 0) of the front SeekHead's
    /// 8-byte Cues `SeekPosition` payload; set when the header is written.
    cues_patch_offset: Option<usize>,
    /// Byte offset of the `Info` `Duration` payload (8 bytes, a float), written
    /// as a placeholder in the two-pass mode and patched at EOS by
    /// [`duration_patch`](Self::duration_patch). `None` in the streaming mode,
    /// which never learns a duration.
    duration_patch_offset: Option<usize>,
    /// Byte offset of the Segment's data start in the muxed output, the absolute
    /// anchor the `segment_pos` positions are relative to. Its own 8-byte size
    /// field is the 8 bytes ahead of it. Meaningful once the header is written.
    segment_data_start: usize,
    /// The reserved size field of each Cluster, in write order, for
    /// [`finalize_seekable`] to fill in. Empty in the streaming mode.
    cluster_size_patches: Vec<SizePatch>,
    /// Highest block end seen, in TimestampScale ticks: the presentation
    /// duration. Per track, the last block's timestamp plus its own duration,
    /// each rounded to a tick the way the block timestamps are, which is how
    /// ffmpeg arrives at the value it writes.
    max_end_ticks: u64,
    /// The previous block's timestamp per track (ticks), so a frame that
    /// declares no duration can borrow the last inter-frame gap.
    prev_ts_ticks: Vec<Option<u64>>,
}

impl MatroskaMuxer {
    /// A single-track muxer (no `CodecPrivate`, the codecs that need none).
    pub fn new(spec: MkvTrackSpec) -> Self {
        Self::new_multi(alloc::vec![MkvTrackConfig {
            spec,
            codec_private: Vec::new()
        }])
    }

    /// A multi-track muxer: the A/V case. Input order is track order; a track's
    /// `CodecPrivate` (empty for none) rides the Tracks element. Clusters span the
    /// shared timeline, so blocks of every track interleave by timestamp.
    pub fn new_multi(tracks: Vec<MkvTrackConfig>) -> Self {
        assert!(!tracks.is_empty(), "MatroskaMuxer needs at least one track");
        // Cues index the video keyframes: pick the first video track, else track 0.
        let cue_track = tracks
            .iter()
            .position(|t| t.spec.codec.track_type() == 1)
            .unwrap_or(0);
        let track_count = tracks.len();
        Self {
            tracks,
            tags: TagList::new(),
            track_tags: Vec::new(),
            chapters: Vec::new(),
            max_cluster_span_ms: DEFAULT_MAX_CLUSTER_SPAN_MS,
            header_written: false,
            cluster_base_ms: None,
            cue_track,
            segment_pos: 0,
            current_cluster_pos: 0,
            cues: Vec::new(),
            last_cued_cluster_pos: None,
            record_cues: true,
            two_pass: false,
            cues_patch_offset: None,
            duration_patch_offset: None,
            segment_data_start: 0,
            cluster_size_patches: Vec::new(),
            max_end_ticks: 0,
            prev_ts_ticks: alloc::vec![None; track_count],
        }
    }

    /// The two-pass / seekable mode: a front `SeekHead`, a reserved `Duration`,
    /// and definite-size Segment and Clusters (see the field note). The caller
    /// must finish the buffered file with [`finalize_seekable`], which fills all
    /// of them in.
    pub fn with_two_pass(mut self) -> Self {
        self.two_pass = true;
        self
    }

    /// Live mode: do not collect the `Cues` entries, for a caller that never
    /// writes the index ([`finish`](Self::finish) then returns nothing). The
    /// output bytes are the same; what changes is that the muxer keeps no
    /// per-Cluster state for a stream with no end.
    pub fn without_cues(mut self) -> Self {
        self.record_cues = false;
        self
    }

    /// Attach stream metadata, written as a `Tags` element after Tracks on the
    /// first frame (the inverse of [`MatroskaDemuxer::tags`]).
    pub fn with_tags(mut self, tags: TagList) -> Self {
        self.tags = tags;
        self
    }

    /// Attach metadata scoped to one track (0-based, the same index
    /// [`push_frame_on`](Self::push_frame_on) takes): written in the same `Tags`
    /// element as a `Tag` whose `Targets` carries that track's `TagTrackUID`, so
    /// a reader attaches it to that elementary stream (M787). Ignored for an
    /// out-of-range index; calling it twice for a track writes both `Tag`s.
    pub fn with_track_tags(mut self, track: usize, tags: TagList) -> Self {
        if track < self.tracks.len() {
            self.track_tags.push((track, tags));
        }
        self
    }

    /// Attach the table of contents, written as a `Chapters` element after
    /// Tracks on the first frame (the inverse of
    /// [`MatroskaDemuxer::chapters`]). Chapter times are stream-time
    /// nanoseconds, the same units the demuxer reports.
    pub fn with_chapters(mut self, chapters: Vec<Chapter>) -> Self {
        self.chapters = chapters;
        self
    }

    /// Cap the time span of one Cluster (ms); a frame this far past the Cluster
    /// base opens a new one. Keep it within the i16 block-timestamp range (±32 s).
    pub fn with_max_cluster_span_ms(mut self, span_ms: u64) -> Self {
        self.max_cluster_span_ms = span_ms;
        self
    }

    /// Mux one frame on the first (or only) track, with no declared duration.
    /// The single-track entry point.
    pub fn push_frame(&mut self, data: &[u8], pts_ns: u64, keyframe: bool) -> Vec<u8> {
        self.push_frame_on(0, data, pts_ns, keyframe, 0)
    }

    /// Mux one frame on track `track` (0-based pad index). The first call writes
    /// the EBML header, Segment, Info, and Tracks (plus Tags when present); then a
    /// SimpleBlock for that track, opening a new (unknown-size) Cluster first when
    /// the shared time window is exceeded.
    ///
    /// `duration_ns` is the frame's presentation duration when upstream knows it
    /// (`0` when it does not). It only changes the output when it is *shorter*
    /// than the packet itself, the end-of-stream trim a container like Ogg
    /// carries in its granule: then the block is written as a `BlockGroup` with a
    /// `BlockDuration`, the only way Matroska can say "this packet ends early"
    /// (M792). Every other frame stays a bare SimpleBlock.
    pub fn push_frame_on(
        &mut self,
        track: usize,
        data: &[u8],
        pts_ns: u64,
        keyframe: bool,
        duration_ns: u64,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        if !self.header_written {
            // WebM only when every track is a WebM-subset codec, else matroska.
            let all_webm = self.tracks.iter().all(|t| t.spec.codec.is_webm());
            let doctype: &[u8] = if all_webm { b"webm" } else { b"matroska" };
            out.extend_from_slice(&ebml_header(doctype));
            // The Segment size is 8 bytes either way: all-ones (unknown) for a
            // stream that runs to end of input, a reserved placeholder the
            // two-pass finalize fills in for a file that declares its bounds.
            id_bytes(ID_SEGMENT, &mut out);
            out.extend_from_slice(&UNKNOWN_SIZE_8);
            // Segment data starts here; track positions are anchored to it.
            let seg_data_start = out.len();
            self.segment_data_start = seg_data_start;
            // Only the two-pass mode can fill a Duration in: it is not known
            // until EOS, and a streaming caller has already emitted the header.
            let (info, duration_at) = info_element(self.two_pass);
            let tracks = tracks_element(&self.tracks, &self.track_tags);
            let chapters = chapters_element(&self.chapters);
            // Empty when every per-track tag went into its TrackEntry instead.
            let tags = tags_element(&self.tags, &self.track_tags);
            if self.two_pass {
                // Front SeekHead with fixed-layout entries (21 bytes each), so
                // the Cues position (unknown until EOS) is patchable in place.
                let entries = 3 + usize::from(!chapters.is_empty()) + usize::from(!tags.is_empty());
                let sh_len = (5 + entries * SEEK_ENTRY_LEN) as u64;
                id_bytes(ID_SEEK_HEAD, &mut out);
                out.push(0x80 | (entries * SEEK_ENTRY_LEN) as u8);
                let mut pos = sh_len;
                seek_entry(ID_INFO, pos, &mut out);
                pos += info.len() as u64;
                seek_entry(ID_TRACKS, pos, &mut out);
                pos += tracks.len() as u64;
                if !chapters.is_empty() {
                    seek_entry(ID_CHAPTERS, pos, &mut out);
                    pos += chapters.len() as u64;
                }
                if !tags.is_empty() {
                    seek_entry(ID_TAGS, pos, &mut out);
                }
                // Cues placeholder: the payload is the last 8 bytes of the entry.
                seek_entry(ID_CUES, 0, &mut out);
                self.cues_patch_offset = Some(out.len() - 8);
            }
            self.duration_patch_offset = duration_at.map(|rel| out.len() + rel);
            out.extend_from_slice(&info);
            out.extend_from_slice(&tracks);
            out.extend_from_slice(&chapters);
            out.extend_from_slice(&tags);
            self.segment_pos = (out.len() - seg_data_start) as u64;
            self.header_written = true;
        }
        let ts = pts_ns / DEFAULT_TIMESTAMP_SCALE;
        self.record_track_end(track, ts, duration_ns);
        let need_new_cluster = match self.cluster_base_ms {
            None => true,
            Some(base) => ts < base || ts - base > self.max_cluster_span_ms,
        };
        if need_new_cluster {
            // Open a Cluster: its id, then a size, then the base Timestamp. The
            // streaming shape is a one-byte unknown-size marker, so the next
            // Cluster header (or EOF) ends it and nothing has to be known ahead
            // of the content; the two-pass shape reserves 8 bytes for the
            // finalize to fill in. The Cluster element starts at the current
            // Segment-data position, which a keyframe in it records as its
            // CueClusterPosition, so the wider header is already accounted for.
            self.current_cluster_pos = self.segment_pos;
            // Where this Cluster lands in the muxed output as a whole, which the
            // patch offsets are absolute in: the Segment data start plus how far
            // past it the stream has run.
            let cluster_at = self.segment_data_start + self.segment_pos as usize;
            let before = out.len();
            id_bytes(ID_CLUSTER, &mut out);
            let size_at = cluster_at + (out.len() - before);
            if self.two_pass {
                out.extend_from_slice(&UNKNOWN_SIZE_8);
                self.cluster_size_patches.push(SizePatch {
                    size_at,
                    data_at: size_at + UNKNOWN_SIZE_8.len(),
                });
            } else {
                out.push(0xFF);
            }
            out.extend_from_slice(&elem_vec(ID_TIMESTAMP, &uint_bytes(ts)));
            self.segment_pos += (out.len() - before) as u64;
            self.cluster_base_ms = Some(ts);
        }
        let base = self.cluster_base_ms.expect("set above");
        let rel = (ts as i64 - base as i64) as i16;
        let track_number = (track + 1) as u64;
        let before = out.len();
        match self.trimmed_block(track, data, duration_ns) {
            // A trimmed packet needs a BlockGroup to carry its trim; the Block's
            // flags stay 0 (the keyframe bit is a SimpleBlock field). Both
            // elements are written, as ffmpeg does: `BlockDuration` is what a
            // generic reader understands, `DiscardPadding` what carries the ns
            // the millisecond grid would round away.
            Some((duration, discard_ns)) => {
                let mut trailer = elem_vec(ID_BLOCK_DURATION, &uint_bytes(duration));
                trailer.extend_from_slice(&elem_vec(ID_DISCARD_PADDING, &int_bytes(discard_ns)));
                put_block_group(&mut out, track_number, rel, data, &trailer);
            }
            // A subtitle cue lasts as long as its `BlockDuration` says, and a
            // SimpleBlock has nowhere to put one, so a text block is always a
            // BlockGroup (M898). The Block's flags stay 0: the keyframe bit is a
            // SimpleBlock field.
            None if self.is_subtitle_track(track) => {
                let mut trailer = Vec::new();
                if duration_ns > 0 {
                    let ticks =
                        (duration_ns + DEFAULT_TIMESTAMP_SCALE / 2) / DEFAULT_TIMESTAMP_SCALE;
                    trailer = elem_vec(ID_BLOCK_DURATION, &uint_bytes(ticks));
                }
                put_block_group(&mut out, track_number, rel, data, &trailer);
            }
            None => put_block(&mut out, ID_SIMPLE_BLOCK, track_number, rel, keyframe, data),
        }
        self.segment_pos += (out.len() - before) as u64;
        // Index this Cluster in the Cues if it holds a keyframe on the cue track,
        // at most once per Cluster (the first such keyframe), to bound the index.
        if self.record_cues
            && keyframe
            && track == self.cue_track
            && self.last_cued_cluster_pos != Some(self.current_cluster_pos)
        {
            self.cues.push((ts, self.current_cluster_pos));
            self.last_cued_cluster_pos = Some(self.current_cluster_pos);
        }
        out
    }

    /// Fold a block into the presentation duration: its timestamp plus how long
    /// it lasts, in ticks, keeping the highest end across tracks. The frame's own
    /// duration when upstream timed it (a demuxer knows the container's trim),
    /// else the gap from this track's previous block, which is what a steady
    /// stream's last frame lasts. Both are rounded to a tick, so the value lands
    /// where ffmpeg's does: the millisecond grid is the container's, not ours.
    fn record_track_end(&mut self, track: usize, ts: u64, duration_ns: u64) {
        let prev = self.prev_ts_ticks.get(track).copied().flatten();
        if let Some(slot) = self.prev_ts_ticks.get_mut(track) {
            *slot = Some(ts);
        }
        let ticks = if duration_ns > 0 {
            (duration_ns + DEFAULT_TIMESTAMP_SCALE / 2) / DEFAULT_TIMESTAMP_SCALE
        } else {
            prev.map_or(0, |p| ts.saturating_sub(p))
        };
        self.max_end_ticks = self.max_end_ticks.max(ts.saturating_add(ticks));
    }

    /// The EOS patch the `Info` `Duration` placeholder needs: `(byte offset of
    /// its 8-byte payload in the muxed output, the big-endian float to write)`.
    /// The value is the presentation duration in TimestampScale ticks. `None`
    /// without [`with_two_pass`](Self::with_two_pass) (the streaming mode
    /// writes no placeholder) or before the header was written.
    pub fn duration_patch(&self) -> Option<(usize, [u8; 8])> {
        let offset = self.duration_patch_offset?;
        Some((offset, (self.max_end_ticks as f64).to_be_bytes()))
    }

    /// Whether track `track` is a subtitle track, text or bitmap: its cues last
    /// as long as the `BlockDuration` only a `BlockGroup` can hold.
    fn is_subtitle_track(&self, track: usize) -> bool {
        self.tracks
            .get(track)
            .is_some_and(|t| t.spec.codec.track_type() == 0x11)
    }

    /// The `(BlockDuration in TimestampScale ticks, DiscardPadding in ns)` a
    /// frame needs because its presentation duration is shorter than the packet
    /// it carries, or `None` when the block says nothing the next timestamp does
    /// not. Only Opus is judged: its packet length is readable from the TOC byte,
    /// so a trimmed tail is recognizable, which is exactly the end-of-stream trim
    /// an Ogg granule or an MP4 sample table carries (M792). The shortfall must
    /// reach a whole tick, since a `BlockDuration` that rounds back to the packet
    /// length is noise.
    fn trimmed_block(&self, track: usize, data: &[u8], duration_ns: u64) -> Option<(u64, i64)> {
        if duration_ns == 0 || self.tracks.get(track)?.spec.codec != MkvCodec::Opus {
            return None;
        }
        let samples = u64::from(crate::opusparse::packet_samples(data));
        let packet_ns = samples * 1_000_000_000 / u64::from(crate::opusparse::OPUS_RATE_HZ);
        let discard_ns = packet_ns.checked_sub(duration_ns)?;
        if discard_ns < DEFAULT_TIMESTAMP_SCALE {
            return None;
        }
        // BlockDuration rounds to the nearest tick, as ffmpeg does (a 6.5 ms tail
        // writes 7); DiscardPadding keeps the exact ns beside it.
        let ticks = (duration_ns + DEFAULT_TIMESTAMP_SCALE / 2) / DEFAULT_TIMESTAMP_SCALE;
        Some((ticks, discard_ns as i64))
    }

    /// The `Cues` element for the keyframes muxed so far, to write once at EOS
    /// (after the last Cluster) so the stream is seekable on a read-to-end. Empty
    /// when no keyframe was indexed (no frames, or none on the cue track). The
    /// inverse of [`MatroskaDemuxer`]'s `Cues` parse; positions are relative to the
    /// Segment data start, the anchor `cue_seek_offset` adds back.
    pub fn finish(&self) -> Vec<u8> {
        if self.cues.is_empty() {
            return Vec::new();
        }
        let track_number = (self.cue_track + 1) as u64;
        let mut body = Vec::new();
        for &(time, cluster_pos) in &self.cues {
            let mut positions = elem_vec(ID_CUE_TRACK, &uint_bytes(track_number));
            positions
                .extend_from_slice(&elem_vec(ID_CUE_CLUSTER_POSITION, &uint_bytes(cluster_pos)));
            let mut point = elem_vec(ID_CUE_TIME, &uint_bytes(time));
            point.extend_from_slice(&elem_vec(ID_CUE_TRACK_POSITIONS, &positions));
            body.extend_from_slice(&elem_vec(ID_CUE_POINT, &point));
        }
        elem_vec(ID_CUES, &body)
    }

    /// The EOS patch a front SeekHead needs: `(byte offset of the Cues
    /// `SeekPosition` payload in the muxed output, the position value to write
    /// as an 8-byte big-endian uint)`. The Cues land where the stream ended, so
    /// call this at EOS, before appending [`finish`](Self::finish)'s bytes.
    /// `None` without [`with_two_pass`](Self::with_two_pass) (or before the
    /// header was written).
    pub fn seek_head_patch(&self) -> Option<(usize, u64)> {
        self.cues_patch_offset.map(|off| (off, self.segment_pos))
    }

    /// The reserved size field of each Cluster, in write order. A Cluster runs to
    /// the next one's header, the last to the end of what was muxed. Empty in the
    /// streaming mode, whose Clusters declare no size.
    pub fn cluster_size_patches(&self) -> &[SizePatch] {
        &self.cluster_size_patches
    }

    /// The reserved Segment size field, which holds everything after the Segment
    /// header, the trailing `Cues` included. `None` in the streaming mode (the
    /// Segment runs to end of stream) or before the header was written.
    pub fn segment_size_patch(&self) -> Option<SizePatch> {
        (self.two_pass && self.header_written).then(|| SizePatch {
            size_at: self.segment_data_start - UNKNOWN_SIZE_8.len(),
            data_at: self.segment_data_start,
        })
    }
}

/// Fill a reserved 8-byte size field with the length its element turned out to
/// have: the widest EBML size form, so the value is written in place.
fn write_size_patch(file: &mut [u8], patch: SizePatch, end: usize) {
    let len = (end - patch.data_at) as u64;
    let mut field = [0x01u8; 8];
    field[1..].copy_from_slice(&len.to_be_bytes()[1..]);
    file[patch.size_at..patch.size_at + 8].copy_from_slice(&field);
}

/// Finalize a buffered two-pass file in place: fill the `Info` `Duration`
/// placeholder with the presentation length (M794) and each Cluster's reserved
/// size with what it turned out to hold (M828), then append the `Cues`, point the
/// front `SeekHead` at where they landed (M770), and close the Segment over the
/// lot. The buffered-output half of [`MatroskaMuxer::with_two_pass`], shared by
/// the single- and multi-track muxer elements; `file` must be everything the muxer
/// emitted, from byte 0, since every patch offset is absolute.
pub fn finalize_seekable(mux: &MatroskaMuxer, file: &mut Vec<u8>) {
    if let Some((off, value)) = mux.duration_patch() {
        file[off..off + 8].copy_from_slice(&value);
    }
    // Each Cluster ends where the next one's id begins (the 4 bytes ahead of its
    // size field), the last where the muxed bytes end: the Cues appended below
    // are the Segment's next child, not part of any Cluster.
    let clusters = mux.cluster_size_patches();
    for (i, patch) in clusters.iter().enumerate() {
        let end = clusters
            .get(i + 1)
            .map_or(file.len(), |next| next.size_at - 4);
        write_size_patch(file, *patch, end);
    }
    let cues = mux.finish();
    if !cues.is_empty() {
        if let Some((off, pos)) = mux.seek_head_patch() {
            file[off..off + 8].copy_from_slice(&pos.to_be_bytes());
        }
        file.extend_from_slice(&cues);
    }
    if let Some(patch) = mux.segment_size_patch() {
        let end = file.len();
        write_size_patch(file, patch, end);
    }
}

/// Byte length of one fixed-layout `Seek` entry written by [`seek_entry`].
const SEEK_ENTRY_LEN: usize = 21;

/// One fixed-layout `Seek` entry: a `SeekID` carrying the target's 4-byte
/// element id and a `SeekPosition` as a fixed 8-byte uint (minimal encoding
/// would vary with the value, and the Cues entry must be patchable in place).
fn seek_entry(target_id: u32, pos: u64, out: &mut Vec<u8>) {
    id_bytes(ID_SEEK, out);
    out.push(0x80 | 18);
    id_bytes(ID_SEEK_ID, out);
    out.push(0x80 | 4);
    out.extend_from_slice(&target_id.to_be_bytes());
    id_bytes(ID_SEEK_POSITION, out);
    out.push(0x80 | 8);
    out.extend_from_slice(&pos.to_be_bytes());
}

/// A minimal but valid EBML header naming the DocType (`matroska` / `webm`).
fn ebml_header(doctype: &[u8]) -> Vec<u8> {
    let mut h = elem_vec(0x4286, &[1]); // EBMLVersion
    h.extend_from_slice(&elem_vec(0x42F7, &[1])); // EBMLReadVersion
    h.extend_from_slice(&elem_vec(0x42F2, &[4])); // EBMLMaxIDLength
    h.extend_from_slice(&elem_vec(0x42F3, &[8])); // EBMLMaxSizeLength
    h.extend_from_slice(&elem_vec(0x4282, doctype)); // DocType
    h.extend_from_slice(&elem_vec(0x4287, &[DOC_TYPE_VERSION])); // DocTypeVersion
    h.extend_from_slice(&elem_vec(0x4285, &[DOC_TYPE_READ_VERSION])); // DocTypeReadVersion
    elem_vec(ID_EBML, &h)
}

/// The `Info` element, and the offset of its `Duration` payload within it when
/// one was reserved. The `Duration` is an 8-byte float in TimestampScale units,
/// written as a zero placeholder because the total is only known at EOS; a
/// two-pass caller patches it through
/// [`MatroskaMuxer::duration_patch`](MatroskaMuxer::duration_patch).
fn info_element(with_duration: bool) -> (Vec<u8>, Option<usize>) {
    let mut body = elem_vec(ID_TIMESTAMP_SCALE, &uint_bytes(DEFAULT_TIMESTAMP_SCALE));
    body.extend_from_slice(&elem_vec(ID_MUXING_APP, APP_NAME.as_bytes()));
    body.extend_from_slice(&elem_vec(ID_WRITING_APP, APP_NAME.as_bytes()));
    let mut payload_at = None;
    if with_duration {
        body.extend_from_slice(&elem_vec(ID_DURATION, &0f64.to_be_bytes()));
        payload_at = Some(body.len() - 8);
    }
    let info = elem_vec(ID_INFO, &body);
    // The element's own header sits ahead of the body inside `info`.
    let header = info.len() - body.len();
    (info, payload_at.map(|at| at + header))
}

/// The `Tracks` element: one `TrackEntry` per track, numbered 1.. in order, each
/// with a `TrackUID` equal to its number (any stable nonzero value is valid, and
/// it is what a `Tags` `Targets` scopes a per-track tag to). A non-empty
/// `CodecPrivate` (avcC / hvcC record, AAC AudioSpecificConfig) is written after
/// the CodecID. A track's [`Tag::Title`] / [`Tag::Language`] ride the entry's own
/// `Name` / `Language` elements (M788), where ffmpeg and every player look for
/// them; the rest of its tags go to the `Tags` element.
fn tracks_element(tracks: &[MkvTrackConfig], track_tags: &[(usize, TagList)]) -> Vec<u8> {
    let mut entries = Vec::new();
    for (i, track) in tracks.iter().enumerate() {
        let spec = &track.spec;
        let codec_id = spec.codec.codec_id().unwrap_or(b"");
        let mut entry = elem_vec(ID_TRACK_NUMBER, &uint_bytes(i as u64 + 1));
        entry.extend_from_slice(&elem_vec(ID_TRACK_UID, &uint_bytes(track_uid(i))));
        entry.extend_from_slice(&elem_vec(
            ID_TRACK_TYPE,
            &uint_bytes(spec.codec.track_type() as u64),
        ));
        if let Some(name) = track_entry_string(track_tags, i, |t| match t {
            Tag::Title(v) => Some(v),
            _ => None,
        }) {
            entry.extend_from_slice(&elem_vec(ID_TRACK_NAME, name.as_bytes()));
        }
        // Matroska's Language is ISO 639-2; the value is written as given, the
        // caller owns its form.
        if let Some(lang) = track_entry_string(track_tags, i, |t| match t {
            Tag::Language(v) => Some(v),
            _ => None,
        }) {
            entry.extend_from_slice(&elem_vec(ID_LANGUAGE, lang.as_bytes()));
        }
        entry.extend_from_slice(&elem_vec(ID_CODEC_ID, codec_id));
        if !track.codec_private.is_empty() {
            entry.extend_from_slice(&elem_vec(ID_CODEC_PRIVATE, &track.codec_private));
        }
        if spec.codec == MkvCodec::Opus {
            // The encoder delay the decoder must discard, as ns rather than the
            // header's 48 kHz samples, plus the seek pre-roll the Matroska Opus
            // mapping mandates (M792).
            entry.extend_from_slice(&elem_vec(
                ID_CODEC_DELAY,
                &uint_bytes(opus_codec_delay_ns(&track.codec_private)),
            ));
            entry.extend_from_slice(&elem_vec(
                ID_SEEK_PRE_ROLL,
                &uint_bytes(OPUS_SEEK_PRE_ROLL_NS),
            ));
        }
        // A subtitle TrackEntry has neither child: `Video` / `Audio` describe
        // settings a text track has none of, and a reader rejects the ones it
        // does not expect for the TrackType.
        match spec.codec.track_type() {
            1 => {
                let mut v = elem_vec(ID_PIXEL_WIDTH, &uint_bytes(spec.width as u64));
                v.extend_from_slice(&elem_vec(ID_PIXEL_HEIGHT, &uint_bytes(spec.height as u64)));
                entry.extend_from_slice(&elem_vec(ID_VIDEO, &v));
            }
            2 => {
                let mut a = elem_vec(ID_CHANNELS, &uint_bytes(spec.channels.max(1) as u64));
                a.extend_from_slice(&elem_vec(
                    ID_SAMPLING_FREQ,
                    &(spec.sample_rate as f64).to_be_bytes(),
                ));
                entry.extend_from_slice(&elem_vec(ID_AUDIO, &a));
            }
            _ => {}
        }
        entries.extend_from_slice(&elem_vec(ID_TRACK_ENTRY, &entry));
    }
    elem_vec(ID_TRACKS, &entries)
}

/// `SeekPreRoll` for an Opus track: the 80 ms the Matroska Opus mapping fixes as
/// the audio a decoder must run through before a seek target to be at full
/// quality. Every writer (ffmpeg included) uses this exact value.
const OPUS_SEEK_PRE_ROLL_NS: u64 = 80_000_000;

/// `CodecDelay` for an Opus track: the `CodecPrivate` `OpusHead`'s pre-skip in
/// ns (the header counts 48 kHz samples). `0` when the header is missing or
/// malformed, which is what a reader assumes anyway.
fn opus_codec_delay_ns(codec_private: &[u8]) -> u64 {
    let Some((_, pre_skip)) = crate::opusparse::parse_opus_head(codec_private) else {
        return 0;
    };
    u64::from(pre_skip) * 1_000_000_000 / u64::from(crate::opusparse::OPUS_RATE_HZ)
}

/// The `TrackUID` the muxer writes for the `i`-th track: its track number, so a
/// `Targets` referring to it needs nothing but the pad index.
fn track_uid(index: usize) -> u64 {
    index as u64 + 1
}

/// The first value `pick` matches among track `index`'s tags: the `Name` /
/// `Language` a `TrackEntry` carries instead of a `SimpleTag` (M788).
fn track_entry_string(
    track_tags: &[(usize, TagList)],
    index: usize,
    pick: fn(&Tag) -> Option<&String>,
) -> Option<&str> {
    track_tags
        .iter()
        .filter(|(i, _)| *i == index)
        .flat_map(|(_, list)| list.tags())
        .find_map(pick)
        .map(String::as_str)
}

/// True for a tag the `TrackEntry` carries itself, so the `Tags` element skips it
/// (no double-write).
fn is_track_entry_tag(tag: &Tag) -> bool {
    matches!(tag, Tag::Title(_) | Tag::Language(_))
}

/// The `Tags` element: the whole-stream tags as one `Tag` with an empty
/// `Targets`, then one `Tag` per tagged track whose `Targets` carries that
/// track's `TagTrackUID` (M787). The inverse of [`parse_tags`]; the typed keys
/// write their conventional uppercase Matroska names. A track's title / language
/// are skipped here, the `TrackEntry` carries them (M788).
fn tags_element(tags: &TagList, track_tags: &[(usize, TagList)]) -> Vec<u8> {
    let mut body = Vec::new();
    let whole_tags: Vec<&Tag> = tags.text_tags().collect();
    if !whole_tags.is_empty() {
        let mut whole = elem_vec(ID_TARGETS, &[]);
        for t in whole_tags {
            whole.extend_from_slice(&simple_tag(t));
        }
        body.extend_from_slice(&elem_vec(ID_TAG, &whole));
    }
    for (index, list) in track_tags {
        let scoped: Vec<&Tag> = list
            .text_tags()
            .filter(|t| !is_track_entry_tag(t))
            .collect();
        if scoped.is_empty() {
            continue;
        }
        let uid = uint_bytes(track_uid(*index));
        let mut simple = Vec::new();
        for t in scoped {
            simple.extend_from_slice(&simple_tag(t));
        }
        let mut tag = elem_vec(ID_TARGETS, &elem_vec(ID_TAG_TRACK_UID, &uid));
        tag.extend_from_slice(&simple);
        body.extend_from_slice(&elem_vec(ID_TAG, &tag));
    }
    if body.is_empty() {
        return Vec::new();
    }
    elem_vec(ID_TAGS, &body)
}

/// The `Chapters` element (M1046): one default `EditionEntry` holding every
/// top-level chapter as a `ChapterAtom`, the inverse of [`parse_chapters`].
/// Empty when there are no chapters, so nothing is written.
fn chapters_element(chapters: &[Chapter]) -> Vec<u8> {
    if chapters.is_empty() {
        return Vec::new();
    }
    let mut edition = elem_vec(ID_EDITION_FLAG_DEFAULT, &uint_bytes(1));
    let mut next_uid = 0u64;
    for chapter in chapters {
        edition.extend_from_slice(&chapter_atom(chapter, 0, &mut next_uid));
    }
    elem_vec(ID_CHAPTERS, &elem_vec(ID_EDITION_ENTRY, &edition))
}

/// One `ChapterAtom`: its UID, its times in nanoseconds, a `ChapterDisplay` for
/// the title, then its nested atoms. `ChapterUID` is mandatory and must be
/// non-zero, so they are numbered from 1 in document order. `ChapLanguage` is
/// written only when the chapter names one, leaving the spec default otherwise.
/// Nesting is bounded like the read side: a caller's deeper atoms are dropped
/// rather than recursed into.
fn chapter_atom(chapter: &Chapter, depth: u32, next_uid: &mut u64) -> Vec<u8> {
    *next_uid += 1;
    let mut body = elem_vec(ID_CHAPTER_UID, &uint_bytes(*next_uid));
    body.extend_from_slice(&elem_vec(
        ID_CHAPTER_TIME_START,
        &uint_bytes(chapter.start_ns),
    ));
    if let Some(end_ns) = chapter.end_ns {
        body.extend_from_slice(&elem_vec(ID_CHAPTER_TIME_END, &uint_bytes(end_ns)));
    }
    if !chapter.title.is_empty() {
        let mut display = elem_vec(ID_CHAP_STRING, chapter.title.as_bytes());
        if let Some(language) = &chapter.language {
            display.extend_from_slice(&elem_vec(ID_CHAP_LANGUAGE, language.as_bytes()));
        }
        body.extend_from_slice(&elem_vec(ID_CHAPTER_DISPLAY, &display));
    }
    if depth < MAX_CHAPTER_DEPTH {
        for sub in &chapter.sub_chapters {
            body.extend_from_slice(&chapter_atom(sub, depth + 1, next_uid));
        }
    }
    elem_vec(ID_CHAPTER_ATOM, &body)
}

/// One `SimpleTag`: the tag's Matroska `TagName` and its `TagString`.
fn simple_tag(tag: &Tag) -> Vec<u8> {
    let (name, value) = tag_name_value(tag);
    let mut simple = elem_vec(ID_TAG_NAME, name.as_bytes());
    simple.extend_from_slice(&elem_vec(ID_TAG_STRING, value.as_bytes()));
    elem_vec(ID_SIMPLE_TAG, &simple)
}

/// A tag's Matroska `TagName` / `TagString` pair. Typed keys use the conventional
/// uppercase names so they round-trip back to the same variant through
/// [`Tag::from_key_value`]; [`Tag::Other`] keeps its stored key, and the integer
/// / freeform variants flatten to the key and decimal value Matroska's
/// string-only `SimpleTag` can carry.
fn tag_name_value(tag: &Tag) -> (Cow<'_, str>, Cow<'_, str>) {
    let name = match tag {
        Tag::Title(_) => Cow::Borrowed("TITLE"),
        Tag::Artist(_) => Cow::Borrowed("ARTIST"),
        Tag::Album(_) => Cow::Borrowed("ALBUM"),
        Tag::Encoder(_) => Cow::Borrowed("ENCODER"),
        Tag::Language(_) => Cow::Borrowed("LANGUAGE"),
        Tag::Comment(_) => Cow::Borrowed("COMMENT"),
        _ => tag.key(),
    };
    (name, tag.value_string())
}

/// Append a `Block` or `SimpleBlock` element: track-number VINT, signed relative
/// timestamp, flags, data.
fn put_block(out: &mut Vec<u8>, id: u32, track: u64, rel: i16, keyframe: bool, data: &[u8]) {
    let mut header = encode_vint(track);
    header.extend_from_slice(&rel.to_be_bytes());
    header.push(if keyframe { 0x80 } else { 0x00 }); // keyframe flag, no lacing
    let body_len = header.len() + data.len();
    out.reserve(elem_len(id, body_len));
    put_elem_header(out, id, body_len);
    out.extend_from_slice(&header);
    out.extend_from_slice(data);
}

/// Append a `BlockGroup` of one flagless `Block` followed by the `trailer`
/// elements.
fn put_block_group(out: &mut Vec<u8>, track: u64, rel: i16, data: &[u8], trailer: &[u8]) {
    let block_body_len = encode_vint(track).len() + BLOCK_TIMESTAMP_AND_FLAGS_LEN + data.len();
    put_elem_header(
        out,
        ID_BLOCK_GROUP,
        elem_len(ID_BLOCK, block_body_len) + trailer.len(),
    );
    put_block(out, ID_BLOCK, track, rel, false, data);
    out.extend_from_slice(trailer);
}

/// A block's 16-bit relative timestamp plus its flags byte.
const BLOCK_TIMESTAMP_AND_FLAGS_LEN: usize = 3;

/// One EBML element: serialized id, a size VINT, then the body.
fn elem_vec(id: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(elem_len(id, body.len()));
    put_elem_header(&mut out, id, body.len());
    out.extend_from_slice(body);
    out
}

/// Append an element's id and the size VINT of a `body_len` body.
fn put_elem_header(out: &mut Vec<u8>, id: u32, body_len: usize) {
    id_bytes(id, out);
    out.extend_from_slice(&encode_vint(body_len as u64));
}

/// A whole element's length: id, size VINT and body.
fn elem_len(id: u32, body_len: usize) -> usize {
    let mut header = Vec::new();
    put_elem_header(&mut header, id, body_len);
    header.len() + body_len
}

/// Serialize an element ID to its 1..4 bytes (the length marker is part of the
/// value, so the byte count follows the highest non-zero byte).
fn id_bytes(id: u32, out: &mut Vec<u8>) {
    let len = if id > 0x00FF_FFFF {
        4
    } else if id > 0x0000_FFFF {
        3
    } else if id > 0x0000_00FF {
        2
    } else {
        1
    };
    for i in (0..len).rev() {
        out.push((id >> (8 * i)) as u8);
    }
}

/// Encode an EBML size as a minimal VINT, avoiding the all-ones (unknown-size)
/// pattern by growing to a longer encoding (the inverse of [`read_size`]).
fn encode_vint(value: u64) -> Vec<u8> {
    let mut len = 1usize;
    while len < 8 && value >= (1u64 << (7 * len)) - 1 {
        len += 1;
    }
    let mut out = alloc::vec![0u8; len];
    let mut v = value;
    for i in (0..len).rev() {
        out[i] = (v & 0xFF) as u8;
        v >>= 8;
    }
    out[0] |= 1 << (8 - len);
    out
}

/// Minimal big-endian two's-complement signed integer element body, the inverse
/// of [`read_int`]. A leading byte is kept when the value would otherwise change
/// sign (13500000 writes as four bytes, not three).
fn int_bytes(v: i64) -> Vec<u8> {
    let bytes = v.to_be_bytes();
    let pad = if v < 0 { 0xFF } else { 0x00 };
    let mut start = 0;
    while start < 7 && bytes[start] == pad && (bytes[start + 1] & 0x80 == pad & 0x80) {
        start += 1;
    }
    bytes[start..].to_vec()
}

/// Minimal big-endian unsigned integer element body (`0` is one zero byte).
fn uint_bytes(v: u64) -> Vec<u8> {
    if v == 0 {
        return alloc::vec![0];
    }
    let bytes = v.to_be_bytes();
    let start = bytes.iter().position(|&b| b != 0).unwrap_or(7);
    bytes[start..].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn a_nul_padded_codec_id_still_names_its_codec() {
        assert_eq!(
            MkvCodec::from_codec_id(b"V_MPEG4/ISO/AVC\0"),
            MkvCodec::H264
        );
        assert_eq!(MkvCodec::from_codec_id(b"A_OPUS\0\0"), MkvCodec::Opus);
    }

    /// Encode `value` as a minimal-length EBML VINT (used for element sizes and
    /// block track numbers in the synthetic builders).
    fn vint(value: u64) -> Vec<u8> {
        // Grow until the value fits and isn't the all-ones (unknown-size) pattern,
        // which a definite size must avoid by using a longer encoding.
        let mut len = 1usize;
        while len < 8 && value >= (1u64 << (7 * len)) - 1 {
            len += 1;
        }
        let mut out = vec![0u8; len];
        let mut v = value;
        for i in (0..len).rev() {
            out[i] = (v & 0xFF) as u8;
            v >>= 8;
        }
        out[0] |= 1 << (8 - len);
        out
    }

    /// An EBML element: raw id bytes, a size VINT, then the body.
    fn elem(id: &[u8], body: &[u8]) -> Vec<u8> {
        let mut out = id.to_vec();
        out.extend_from_slice(&vint(body.len() as u64));
        out.extend_from_slice(body);
        out
    }

    /// Minimal big-endian unsigned element body.
    fn uint_body(v: u64) -> Vec<u8> {
        if v == 0 {
            return vec![0];
        }
        let mut bytes = v.to_be_bytes().to_vec();
        while bytes.len() > 1 && bytes[0] == 0 {
            bytes.remove(0);
        }
        bytes
    }

    /// A (Simple)Block body: track VINT, signed rel timestamp, flags, frame.
    fn block_body(track: u64, rel: i16, keyframe: bool, frame: &[u8]) -> Vec<u8> {
        let mut b = vint(track);
        b.extend_from_slice(&rel.to_be_bytes());
        b.push(if keyframe { 0x80 } else { 0x00 });
        b.extend_from_slice(frame);
        b
    }

    fn track_entry(
        number: u64,
        codec: &[u8],
        video: Option<(u32, u32)>,
        audio: Option<(u8, u32)>,
    ) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&elem(&[0xD7], &uint_body(number)));
        body.extend_from_slice(&elem(&[0x86], codec));
        if let Some((w, h)) = video {
            let v = [
                elem(&[0xB0], &uint_body(w as u64)),
                elem(&[0xBA], &uint_body(h as u64)),
            ]
            .concat();
            body.extend_from_slice(&elem(&[0xE0], &v));
        }
        if let Some((ch, sr)) = audio {
            let mut a = elem(&[0x9F], &uint_body(ch as u64));
            a.extend_from_slice(&elem(&[0xB5], &(sr as f32).to_be_bytes()));
            body.extend_from_slice(&elem(&[0xE1], &a));
        }
        elem(&[0xAE], &body)
    }

    /// A full single-segment WebM: EBML header, Tracks (VP9 video + Opus audio),
    /// one Cluster with three blocks (two video, one audio).
    fn synthetic_webm() -> Vec<u8> {
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &[
                track_entry(1, b"V_VP9", Some((640, 480)), None),
                track_entry(2, b"A_OPUS", None, Some((2, 48_000))),
            ]
            .concat(),
        );
        let cluster = elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[
                elem(&[0xE7], &uint_body(1000)), // cluster timestamp
                elem(&[0xA3], &block_body(1, 0, true, &[0xDE, 0xAD])),
                elem(&[0xA3], &block_body(2, 0, true, &[0xBE, 0xEF])),
                elem(&[0xA3], &block_body(1, 33, false, &[0xCA, 0xFE])),
            ]
            .concat(),
        );
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &[tracks, cluster].concat());
        [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat()
    }

    /// An `A_FLAC` track parses to `MkvCodec::Flac` with its `CodecPrivate`
    /// (the native `fLaC` STREAMINFO header) retrievable per track number, and an
    /// `A_AC3` track (incl. a BSID-suffixed CodecID) parses to `MkvCodec::Ac3`
    /// with no private data. Drives M757's decoder extradata forwarding.
    #[test]
    fn parses_ac3_and_flac_tracks_with_codec_private() {
        // track_entry has no CodecPrivate arg; build the FLAC entry by hand.
        let flac_private = b"fLaC\x00\x00\x00\x22streaminfo-bytes";
        let mut flac_body = Vec::new();
        flac_body.extend_from_slice(&elem(&[0xD7], &uint_body(1)));
        flac_body.extend_from_slice(&elem(&[0x86], b"A_FLAC"));
        flac_body.extend_from_slice(&elem(&[0x63, 0xA2], flac_private));
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &[
                elem(&[0xAE], &flac_body),
                track_entry(2, b"A_AC3/BSID", None, Some((2, 48_000))),
            ]
            .concat(),
        );
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &tracks);
        let mut d = MatroskaDemuxer::new();
        d.push_data(&[elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat());

        assert_eq!(d.tracks()[0].codec, MkvCodec::Flac);
        assert_eq!(d.codec_private(1), Some(&flac_private[..]));
        assert_eq!(d.tracks()[1].codec, MkvCodec::Ac3);
        assert_eq!(d.codec_private(2), None, "AC-3 carries no private data");
    }

    #[test]
    fn a_vfw_track_maps_by_its_bitmapinfoheader_fourcc() {
        const BITMAPINFO_LEN: usize = 40;
        let vfw_entry = |number: u64, fourcc: &[u8; 4]| {
            let mut bitmap_info = vec![0u8; BITMAPINFO_LEN];
            bitmap_info[16..20].copy_from_slice(fourcc);
            let body = [
                elem(&[0xD7], &uint_body(number)),
                elem(&[0x86], b"V_MS/VFW/FOURCC"),
                elem(&[0x63, 0xA2], &bitmap_info),
            ]
            .concat();
            elem(&[0xAE], &body)
        };
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &[vfw_entry(1, b"MJPG"), vfw_entry(2, b"XVID")].concat(),
        );
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &tracks);
        let mut d = MatroskaDemuxer::new();
        d.push_data(&[elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat());

        assert_eq!(d.tracks()[0].codec, MkvCodec::Mjpeg);
        #[cfg(not(feature = "ffmpeg-long-tail"))]
        {
            assert_eq!(d.tracks()[1].codec, MkvCodec::Other);
            assert_eq!(d.unmapped_codec_ids(), ["V_MS/VFW/FOURCC XVID"]);
        }
        #[cfg(feature = "ffmpeg-long-tail")]
        {
            let xvid = CodecTag {
                namespace: TagNamespace::Riff,
                tag: u32::from_le_bytes(*b"XVID"),
            };
            assert_eq!(d.tracks()[1].codec, MkvCodec::TaggedVideo(xvid));
            assert!(d.unmapped_codec_ids().is_empty());
        }
    }

    #[cfg(feature = "ffmpeg-long-tail")]
    #[test]
    fn acm_and_quicktime_tracks_go_on_under_their_tags() {
        const WAVEFORMATEX_LEN: usize = 18;
        const VISUAL_SAMPLE_ENTRY_LEN: usize = 86;
        const TRUESPEECH: u16 = 0x0022;
        let entry = |number: u64, codec_id: &[u8], private: &[u8]| {
            let body = [
                elem(&[0xD7], &uint_body(number)),
                elem(&[0x86], codec_id),
                elem(&[0x63, 0xA2], private),
            ]
            .concat();
            elem(&[0xAE], &body)
        };
        let mut wave_format = vec![0u8; WAVEFORMATEX_LEN];
        wave_format[..2].copy_from_slice(&TRUESPEECH.to_le_bytes());
        let mut sample_entry = vec![0u8; VISUAL_SAMPLE_ENTRY_LEN];
        sample_entry[4..8].copy_from_slice(b"rle ");
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &[
                entry(1, b"A_MS/ACM", &wave_format),
                entry(2, b"V_QUICKTIME", &sample_entry),
            ]
            .concat(),
        );
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &tracks);
        let mut d = MatroskaDemuxer::new();
        d.push_data(&[elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat());
        assert_eq!(
            d.tracks()[0].codec,
            MkvCodec::TaggedAudio(CodecTag {
                namespace: TagNamespace::Riff,
                tag: u32::from(TRUESPEECH),
            })
        );
        assert_eq!(
            d.tracks()[1].codec,
            MkvCodec::TaggedVideo(CodecTag {
                namespace: TagNamespace::Mov,
                tag: u32::from_le_bytes(*b"rle "),
            })
        );
        assert_eq!(d.codec_private(2), Some(sample_entry.as_slice()));
    }

    /// A Matroska `S_TEXT/UTF8` subtitle track is recognized as
    /// `MkvCodec::Subtitle(Utf8)` and its `BlockGroup` cue is demuxed with the
    /// UTF-8 text, the cluster+block PTS, and the `BlockDuration` (scaled) as the
    /// display window, so a subtitle track flows like a timed-text stream.
    #[test]
    fn extracts_a_subtitle_track_with_block_duration() {
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &track_entry(1, b"S_TEXT/UTF8", None, None),
        );
        // A BlockGroup: Block (track 1, rel 0, "Hello") + BlockDuration 2000 ticks.
        let block = elem(&[0xA1], &block_body(1, 0, true, b"Hello"));
        let duration = elem(&[0x9B], &uint_body(2000));
        let group = elem(&[0xA0], &[block, duration].concat());
        let cluster = elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[elem(&[0xE7], &uint_body(1000)), group].concat(), // cluster timestamp 1000
        );
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &[tracks, cluster].concat());
        let file = [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat();

        let mut d = MatroskaDemuxer::new();
        d.push_data(&file);
        assert_eq!(
            d.tracks()[0].codec,
            MkvCodec::Subtitle(TextFormat::Utf8),
            "S_TEXT/UTF8 maps to a Utf8 subtitle track"
        );
        let frames = d.take_frames();
        assert_eq!(frames.len(), 1, "the cue is demuxed");
        let f = &frames[0];
        assert_eq!(f.codec, MkvCodec::Subtitle(TextFormat::Utf8));
        assert_eq!(f.data, b"Hello", "the block payload is the cue text");
        // Default TimestampScale is 1_000_000 ns/tick (no Info element overrides it).
        assert_eq!(
            f.pts_ns,
            1000 * 1_000_000,
            "cluster+block timestamp, scaled"
        );
        assert_eq!(f.duration_ns, 2000 * 1_000_000, "BlockDuration, scaled");
    }

    /// A `ContentEncodings` element declaring one block-scoped compression with
    /// `algo`, plus the `ContentCompSettings` header stripping needs.
    fn content_encodings(algo: u64, settings: &[u8]) -> Vec<u8> {
        let mut comp = elem(&[0x42, 0x54], &uint_body(algo));
        if !settings.is_empty() {
            comp.extend_from_slice(&elem(&[0x42, 0x55], settings));
        }
        elem(
            &[0x6D, 0x80],
            &elem(&[0x62, 0x40], &elem(&[0x50, 0x34], &comp)),
        )
    }

    /// A single-track file whose `TrackEntry` carries the `entry_children`
    /// elements (`ContentEncodings`, `Audio`) and whose one Cluster holds `block`
    /// as that track's only block payload.
    fn encoded_file(codec: &[u8], entry_children: &[u8], block: &[u8]) -> Vec<u8> {
        let mut entry = elem(&[0xD7], &uint_body(1));
        entry.extend_from_slice(&elem(&[0x86], codec));
        entry.extend_from_slice(entry_children);
        let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &elem(&[0xAE], &entry));
        let cluster = elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[
                elem(&[0xE7], &uint_body(0)),
                elem(&[0xA3], &block_body(1, 0, true, block)),
            ]
            .concat(),
        );
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &[tracks, cluster].concat());
        [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat()
    }

    /// zlib (`ContentCompAlgo` 0) blocks are inflated at demux (M910), so
    /// downstream gets the real payload and the track is not flagged. A file of
    /// this shape decodes to the same text under ffmpeg.
    #[test]
    fn inflates_zlib_compressed_blocks() {
        let payload = b"a subtitle cue, zlib-compressed in the file. ".repeat(8);
        let block = miniz_oxide::deflate::compress_to_vec_zlib(&payload, 6);
        assert!(
            block.len() < payload.len(),
            "the fixture is really compressed"
        );

        let mut d = MatroskaDemuxer::new();
        d.push_data(&encoded_file(
            b"S_TEXT/UTF8",
            &content_encodings(0, &[]),
            &block,
        ));
        assert!(!d.tracks()[0].unsupported_encoding, "zlib is undone here");
        let frames = d.take_frames();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data, payload, "the block inflates to the payload");
    }

    /// Header stripping (`ContentCompAlgo` 3, mkvmerge's default for some
    /// tracks): the `ContentCompSettings` bytes are prepended back onto every
    /// block, so the frame is the original access unit again.
    #[test]
    fn restores_header_stripped_blocks() {
        let header = [0xFF, 0xF1, 0x50, 0x80];
        let stored = b"aac frame minus its ADTS header";

        let mut d = MatroskaDemuxer::new();
        d.push_data(&encoded_file(
            b"A_AAC",
            &content_encodings(3, &header),
            stored,
        ));
        assert!(!d.tracks()[0].unsupported_encoding);
        let frames = d.take_frames();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data, [&header[..], stored].concat());
    }

    /// A block that is not a zlib stream drops instead of forwarding garbage,
    /// and the demuxer keeps parsing the rest of the file.
    #[test]
    fn drops_a_malformed_zlib_block() {
        let mut d = MatroskaDemuxer::new();
        d.push_data(&encoded_file(
            b"S_TEXT/UTF8",
            &content_encodings(0, &[]),
            b"\x78\x9c definitely not a deflate stream",
        ));
        assert_eq!(d.tracks().len(), 1, "the file still parses");
        assert!(
            d.take_frames().is_empty(),
            "a block that does not inflate is dropped"
        );
    }

    /// A crafted block whose output runs past the inflate bound fails rather
    /// than allocating what the stream asks for.
    #[test]
    fn refuses_an_oversized_inflate() {
        let bomb =
            miniz_oxide::deflate::compress_to_vec_zlib(&vec![0u8; MAX_INFLATED_BLOCK_LEN + 1], 6);
        assert!(bomb.len() < 64 * 1024, "a small block, a huge output");
        assert_eq!(
            ContentCompression::Zlib.decode(&bomb, MAX_INFLATED_BLOCK_LEN),
            None,
            "the bound refuses it"
        );

        let mut d = MatroskaDemuxer::new();
        d.push_data(&encoded_file(
            b"S_TEXT/UTF8",
            &content_encodings(0, &[]),
            &bomb,
        ));
        assert!(d.take_frames().is_empty());
    }

    /// An unknown algorithm, bzip2 (1) / lzo (2) without `matroska-compression`,
    /// a `ContentEncryption` and a scope naming the next encoding are not undone:
    /// the block forwards exactly as the file stored it and the track says so,
    /// which is what the bitmap-subtitle path refuses on.
    #[test]
    fn flags_encodings_it_cannot_undo() {
        #[cfg(not(feature = "matroska-compression"))]
        let algos = [1u64, 2, 4];
        #[cfg(feature = "matroska-compression")]
        let algos = [4u64];
        for algo in algos {
            let mut d = MatroskaDemuxer::new();
            d.push_data(&encoded_file(
                b"S_TEXT/UTF8",
                &content_encodings(algo, &[]),
                b"stored bytes",
            ));
            assert!(d.tracks()[0].unsupported_encoding, "algo {algo}");
            assert_eq!(d.take_frames()[0].data, &b"stored bytes"[..], "algo {algo}");
        }

        // ContentEncodingType 1 with a ContentEncryption, no compression at all.
        let encrypted = elem(
            &[0x6D, 0x80],
            &elem(
                &[0x62, 0x40],
                &[elem(&[0x50, 0x33], &uint_body(1)), elem(&[0x50, 0x35], &[])].concat(),
            ),
        );
        let mut d = MatroskaDemuxer::new();
        d.push_data(&encoded_file(b"S_TEXT/UTF8", &encrypted, b"stored bytes"));
        assert!(d.tracks()[0].unsupported_encoding, "encryption");
        assert_eq!(d.take_frames()[0].data, &b"stored bytes"[..]);

        // scope 4 compresses the next ContentEncoding, which players do not support
        let next_scoped = chained_encodings(&[compression_step(0, 4, 0, &[])]);
        let mut d = MatroskaDemuxer::new();
        d.push_data(&encoded_file(b"S_TEXT/UTF8", &next_scoped, b"stored bytes"));
        assert!(d.tracks()[0].unsupported_encoding, "scope 4");
        assert_eq!(d.take_frames()[0].data, &b"stored bytes"[..]);
    }

    // one compression `ContentEncoding` body: its order, scope and `ContentCompression`
    fn compression_step(order: u64, scope: u64, algo: u64, settings: &[u8]) -> Vec<u8> {
        let mut compression = elem(&[0x42, 0x54], &uint_body(algo));
        if !settings.is_empty() {
            compression.extend_from_slice(&elem(&[0x42, 0x55], settings));
        }
        [
            elem(&[0x50, 0x31], &uint_body(order)),
            elem(&[0x50, 0x32], &uint_body(scope)),
            elem(&[0x50, 0x34], &compression),
        ]
        .concat()
    }

    fn chained_encodings(steps: &[Vec<u8>]) -> Vec<u8> {
        let encodings: Vec<u8> = steps
            .iter()
            .flat_map(|step| elem(&[0x62, 0x40], step))
            .collect();
        elem(&[0x6D, 0x80], &encodings)
    }

    fn codec_private_element(private: &[u8]) -> Vec<u8> {
        elem(&[0x63, 0xA2], private)
    }

    /// RFC 9559 undoes the highest `ContentEncodingOrder` first, whatever order
    /// the elements come in: a block stripped of its header and then deflated
    /// is inflated before the header goes back on.
    #[test]
    fn undoes_a_chain_highest_order_first() {
        let header = b"- ";
        let payload = b"- a subtitle line, header-stripped and then zlib-compressed.";
        let stored = miniz_oxide::deflate::compress_to_vec_zlib(&payload[header.len()..], 6);
        let strip = compression_step(0, SCOPE_BLOCK, 3, header);
        let zlib = compression_step(1, SCOPE_BLOCK, 0, &[]);
        for steps in [[strip.clone(), zlib.clone()], [zlib, strip]] {
            let mut d = MatroskaDemuxer::new();
            d.push_data(&encoded_file(
                b"S_TEXT/UTF8",
                &chained_encodings(&steps),
                &stored,
            ));
            assert!(!d.tracks()[0].unsupported_encoding);
            assert_eq!(d.take_frames()[0].data, payload);
        }
    }

    /// A chain is flagged whole when two steps share an order (no order to undo
    /// them in), when any step is an encryption, or when it is longer than real
    /// files make it.
    #[test]
    fn flags_a_chain_it_cannot_undo() {
        let encryption = [
            elem(&[0x50, 0x31], &uint_body(1)),
            elem(&[0x50, 0x33], &uint_body(1)),
            elem(&[0x50, 0x35], &[]),
        ]
        .concat();
        let too_long: Vec<Vec<u8>> = (0..=MAX_CONTENT_ENCODINGS as u64)
            .map(|order| compression_step(order, SCOPE_BLOCK, 3, b"x"))
            .collect();
        let chains = [
            (
                "repeated order",
                vec![
                    compression_step(0, SCOPE_BLOCK, 3, b"x"),
                    compression_step(0, SCOPE_BLOCK, 0, &[]),
                ],
            ),
            (
                "encryption",
                vec![compression_step(0, SCOPE_BLOCK, 3, b"x"), encryption],
            ),
            ("too long", too_long),
        ];
        for (name, steps) in chains {
            let mut d = MatroskaDemuxer::new();
            d.push_data(&encoded_file(
                b"S_TEXT/UTF8",
                &chained_encodings(&steps),
                b"stored bytes",
            ));
            assert!(d.tracks()[0].unsupported_encoding, "{name}");
            assert_eq!(d.take_frames()[0].data, &b"stored bytes"[..], "{name}");
        }
    }

    /// Scope 2 compresses the `CodecPrivate`: it is inflated at parse, and the
    /// blocks are touched only when the scope also names them (scope 3).
    #[test]
    fn inflates_a_compressed_codec_private() {
        let private = b"[Script Info]\nScriptType: v4.00+\n".repeat(8);
        let block = b"0,0,Default,,0,0,0,,a line";
        let deflated_private = miniz_oxide::deflate::compress_to_vec_zlib(&private, 6);
        let deflated_block = miniz_oxide::deflate::compress_to_vec_zlib(block, 6);
        let cases = [
            (SCOPE_CODEC_PRIVATE, &block[..]),
            (SCOPE_BLOCK | SCOPE_CODEC_PRIVATE, &deflated_block),
        ];
        for (scope, stored_block) in cases {
            let entry = [
                codec_private_element(&deflated_private),
                chained_encodings(&[compression_step(0, scope, 0, &[])]),
            ]
            .concat();
            let mut d = MatroskaDemuxer::new();
            d.push_data(&encoded_file(b"S_TEXT/ASS", &entry, stored_block));
            assert!(!d.tracks()[0].unsupported_encoding, "scope {scope}");
            assert_eq!(d.codec_private(1), Some(&private[..]), "scope {scope}");
            assert_eq!(d.take_frames()[0].data, block, "scope {scope}");
        }
    }

    #[cfg(feature = "matroska-decrypt")]
    const TEST_KEY_ID: [u8; 16] = [7; 16];
    #[cfg(feature = "matroska-decrypt")]
    const TEST_KEY: [u8; 16] = [9; 16];
    #[cfg(feature = "matroska-decrypt")]
    const TEST_IV: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

    // one encryption `ContentEncoding` body, WebM Encryption's AES-CTR unless `cipher_mode` says otherwise
    #[cfg(feature = "matroska-decrypt")]
    fn encryption_step(order: u64, scope: u64, cipher_mode: u64) -> Vec<u8> {
        let encryption = [
            elem(&[0x47, 0xE1], &uint_body(CONTENT_ENC_ALGO_AES)),
            elem(&[0x47, 0xE2], &TEST_KEY_ID),
            elem(&[0x47, 0xE7], &elem(&[0x47, 0xE8], &uint_body(cipher_mode))),
        ]
        .concat();
        [
            elem(&[0x50, 0x31], &uint_body(order)),
            elem(&[0x50, 0x32], &uint_body(scope)),
            elem(&[0x50, 0x33], &uint_body(TYPE_ENCRYPTION)),
            elem(&[0x50, 0x35], &encryption),
        ]
        .concat()
    }

    #[cfg(feature = "matroska-decrypt")]
    fn decrypted_frames(entry_children: &[u8], block: &[u8]) -> (bool, Vec<Vec<u8>>) {
        let mut d = MatroskaDemuxer::new().with_content_keys(&[(TEST_KEY_ID, TEST_KEY)]);
        d.push_data(&encoded_file(b"V_VP9", entry_children, block));
        let flagged = d.tracks()[0].unsupported_encoding;
        (
            flagged,
            d.take_frames().into_iter().map(|f| f.data).collect(),
        )
    }

    #[cfg(feature = "matroska-decrypt")]
    fn partitioned_block(partition_count: u8, offsets: &[u32], frame: &[u8]) -> Vec<u8> {
        let mut block = vec![SIGNAL_ENCRYPTED | SIGNAL_PARTITIONED];
        block.extend_from_slice(&TEST_IV);
        block.push(partition_count);
        for offset in offsets {
            block.extend_from_slice(&offset.to_be_bytes());
        }
        block.extend_from_slice(frame);
        block
    }

    #[cfg(feature = "matroska-decrypt")]
    #[test]
    fn drops_an_encrypted_frame_with_a_malformed_header() {
        let encryption = chained_encodings(&[encryption_step(0, SCOPE_BLOCK, AES_CIPHER_MODE_CTR)]);
        let frame = [0xAA; 8];
        let well_formed = partitioned_block(2, &[2, 4], &frame);
        let (flagged, frames) = decrypted_frames(&encryption, &well_formed);
        assert!(!flagged);
        assert_eq!(frames.len(), 1, "the control frame decrypts");
        assert_eq!(frames[0][..2], frame[..2], "the first section is clear");
        assert_eq!(frames[0][4..], frame[4..], "the last section is clear");
        assert_ne!(
            frames[0][2..4],
            frame[2..4],
            "the middle section is deciphered"
        );

        let mut extended = well_formed.clone();
        extended[0] |= SIGNAL_EXTENDED;
        let truncated_iv = [&[SIGNAL_ENCRYPTED][..], &TEST_IV[..5]].concat();
        let truncated_table = partitioned_block(2, &[2], &[]);
        let descending = partitioned_block(2, &[4, 2], &frame);
        let past_the_frame = partitioned_block(1, &[9], &frame);
        let oversized_count = partitioned_block(u8::MAX, &[2, 4], &frame);
        let cases = [
            ("a second signal byte", extended),
            (
                "partitioned but not encrypted",
                vec![SIGNAL_PARTITIONED, 0xAA],
            ),
            ("a truncated IV", truncated_iv),
            ("a truncated partition table", truncated_table),
            ("descending offsets", descending),
            ("an offset past the frame", past_the_frame),
            ("a partition count past the block", oversized_count),
        ];
        for (name, block) in cases {
            let (flagged, frames) = decrypted_frames(&encryption, &block);
            assert!(!flagged, "{name}");
            assert!(frames.is_empty(), "{name} drops the frame");
        }
    }

    // reserved signal bits are ignored, as the format requires
    #[cfg(feature = "matroska-decrypt")]
    #[test]
    fn a_clear_frame_in_an_encrypted_track_passes_through() {
        let encryption = chained_encodings(&[encryption_step(0, SCOPE_BLOCK, AES_CIPHER_MODE_CTR)]);
        let payload = b"an unencrypted frame";
        for signal in [0x00, 0x10] {
            let (flagged, frames) =
                decrypted_frames(&encryption, &[&[signal][..], payload].concat());
            assert!(!flagged);
            assert_eq!(frames, [payload.to_vec()], "signal {signal:#x}");
        }
    }

    // a CBC cipher mode, an encrypted CodecPrivate and a key id without a key all stay flagged
    #[cfg(feature = "matroska-decrypt")]
    #[test]
    fn flags_an_encryption_it_cannot_undo() {
        let stored = b"stored bytes";
        let cases = [
            ("cbc", encryption_step(0, SCOPE_BLOCK, 2)),
            (
                "codec private scope",
                encryption_step(0, SCOPE_BLOCK | SCOPE_CODEC_PRIVATE, AES_CIPHER_MODE_CTR),
            ),
        ];
        for (name, step) in cases {
            let (flagged, frames) = decrypted_frames(&chained_encodings(&[step]), stored);
            assert!(flagged, "{name}");
            assert_eq!(frames, [stored.to_vec()], "{name}");
        }

        let encryption = chained_encodings(&[encryption_step(0, SCOPE_BLOCK, AES_CIPHER_MODE_CTR)]);
        let mut d = MatroskaDemuxer::new();
        d.push_data(&encoded_file(b"V_VP9", &encryption, stored));
        assert!(d.tracks()[0].unsupported_encoding, "no key");
        assert_eq!(d.take_frames()[0].data, stored);
    }

    // the encryption (order 1) comes off before the header strip (order 0) goes back on
    #[cfg(feature = "matroska-decrypt")]
    #[test]
    fn undoes_an_encryption_in_its_chain_order() {
        let header = b"- ";
        let payload = b"- a subtitle line, header-stripped and then encrypted.";
        let stored = [&[0x00][..], &payload[header.len()..]].concat();
        let strip = compression_step(0, SCOPE_BLOCK, 3, header);
        let encryption = encryption_step(1, SCOPE_BLOCK, AES_CIPHER_MODE_CTR);
        for steps in [[strip.clone(), encryption.clone()], [encryption, strip]] {
            let (flagged, frames) = decrypted_frames(&chained_encodings(&steps), &stored);
            assert!(!flagged);
            assert_eq!(frames, [payload.to_vec()]);
        }
    }

    /// A `CodecPrivate` that does not inflate, or inflates past its bound, is
    /// dropped rather than handed on as the stored bytes. The blocks still demux.
    #[test]
    fn drops_a_codec_private_that_does_not_decode() {
        let oversized = miniz_oxide::deflate::compress_to_vec_zlib(
            &vec![0u8; MAX_INFLATED_CODEC_PRIVATE_LEN + 1],
            6,
        );
        for private in [&b"\x78\x9c not a deflate stream"[..], &oversized] {
            let entry = [
                codec_private_element(private),
                chained_encodings(&[compression_step(0, SCOPE_CODEC_PRIVATE, 0, &[])]),
            ]
            .concat();
            let mut d = MatroskaDemuxer::new();
            d.push_data(&encoded_file(b"S_TEXT/ASS", &entry, b"a line"));
            assert_eq!(d.codec_private(1), None);
            assert_eq!(d.take_frames()[0].data, &b"a line"[..]);
        }
    }

    // Python's bz2 and python-lzo compressed these, in tools/mkv-content-encoding-fixtures.sh
    #[cfg(feature = "matroska-compression")]
    const ENCODING_PAYLOAD: &[u8] = include_bytes!("../tests/fixtures/mkv_encoding_payload.txt");
    #[cfg(feature = "matroska-compression")]
    const ENCODING_PAYLOAD_BZIP2: &[u8] =
        include_bytes!("../tests/fixtures/mkv_encoding_payload.bz2");
    #[cfg(feature = "matroska-compression")]
    const ENCODING_PAYLOAD_LZO: &[u8] =
        include_bytes!("../tests/fixtures/mkv_encoding_payload.lzo");
    #[cfg(feature = "matroska-compression")]
    const ZEROS_PAST_THE_BLOCK_BOUND_BZIP2: &[u8] =
        include_bytes!("../tests/fixtures/mkv_encoding_zeros.bz2");

    /// bzip2 (1) and lzo (2) blocks decode at demux with `matroska-compression`,
    /// through output buffers grown past the first one.
    #[cfg(feature = "matroska-compression")]
    #[test]
    fn decodes_bzip2_and_lzo_blocks() {
        for (algo, stored) in [(1u64, ENCODING_PAYLOAD_BZIP2), (2, ENCODING_PAYLOAD_LZO)] {
            let mut d = MatroskaDemuxer::new();
            d.push_data(&encoded_file(
                b"S_TEXT/ASS",
                &content_encodings(algo, &[]),
                stored,
            ));
            assert!(!d.tracks()[0].unsupported_encoding, "algo {algo}");
            let frames = d.take_frames();
            assert_eq!(frames.len(), 1, "algo {algo}");
            assert_eq!(frames[0].data, ENCODING_PAYLOAD, "algo {algo}");
        }
    }

    /// A truncated or damaged bzip2 / lzo block drops instead of forwarding
    /// garbage. lzo has no checksum, so its damage is a back-reference before
    /// the start of the output.
    #[cfg(feature = "matroska-compression")]
    #[test]
    fn drops_corrupt_bzip2_and_lzo_blocks() {
        let mut flipped_bzip2 = ENCODING_PAYLOAD_BZIP2.to_vec();
        flipped_bzip2[ENCODING_PAYLOAD_BZIP2.len() / 2] ^= 0xFF;
        let lzo_match_before_output = [0x11, 0x04, 0x00];
        let cases: [(u64, &[u8]); 4] = [
            (
                1,
                &ENCODING_PAYLOAD_BZIP2[..ENCODING_PAYLOAD_BZIP2.len() / 2],
            ),
            (1, &flipped_bzip2),
            (2, &ENCODING_PAYLOAD_LZO[..ENCODING_PAYLOAD_LZO.len() / 2]),
            (2, &lzo_match_before_output),
        ];
        for (algo, stored) in cases {
            let mut d = MatroskaDemuxer::new();
            d.push_data(&encoded_file(
                b"S_TEXT/ASS",
                &content_encodings(algo, &[]),
                stored,
            ));
            assert_eq!(d.tracks().len(), 1, "algo {algo}");
            assert!(d.take_frames().is_empty(), "algo {algo}");
        }
    }

    /// Output one byte past the bound fails, output at the bound decodes, and a
    /// 46-byte bzip2 block that expands past the block bound drops at demux.
    #[cfg(feature = "matroska-compression")]
    #[test]
    fn refuses_bzip2_and_lzo_output_past_the_bound() {
        let exact = ENCODING_PAYLOAD.len();
        let cases = [
            (ContentCompression::Bzip2, ENCODING_PAYLOAD_BZIP2),
            (ContentCompression::Lzo, ENCODING_PAYLOAD_LZO),
        ];
        for (step, stored) in cases {
            assert_eq!(step.decode(stored, exact - 1), None, "{step:?}");
            assert_eq!(
                step.decode(stored, exact).as_deref(),
                Some(ENCODING_PAYLOAD),
                "{step:?}"
            );
        }

        let mut d = MatroskaDemuxer::new();
        d.push_data(&encoded_file(
            b"S_TEXT/ASS",
            &content_encodings(1, &[]),
            ZEROS_PAST_THE_BLOCK_BOUND_BZIP2,
        ));
        assert!(d.take_frames().is_empty());
    }

    /// `A_PCM/*` maps to the PCM format of its byte order and `BitDepth`, and a
    /// depth no g2g format carries stays an unmapped codec `decodebin` names.
    #[test]
    fn maps_pcm_codec_ids_at_their_bit_depth() {
        let pcm = |format, big_endian| Some(MkvCodec::Pcm { format, big_endian });
        let cases: [(&[u8], u64, Option<MkvCodec>); 5] = [
            (b"A_PCM/INT/LIT", 8, pcm(AudioFormat::PcmU8, false)),
            (b"A_PCM/INT/LIT", 16, pcm(AudioFormat::PcmS16Le, false)),
            (b"A_PCM/INT/BIG", 24, pcm(AudioFormat::PcmS24Le, true)),
            (b"A_PCM/FLOAT/IEEE", 32, pcm(AudioFormat::PcmF32Le, false)),
            (b"A_PCM/FLOAT/IEEE", 64, None),
        ];
        for (codec_id, bit_depth, want) in cases {
            let audio = elem(&[0xE1], &elem(&[0x62, 0x64], &uint_body(bit_depth)));
            let mut d = MatroskaDemuxer::new();
            d.push_data(&encoded_file(codec_id, &audio, b"samples"));
            let name = core::str::from_utf8(codec_id).unwrap();
            match want {
                Some(codec) => assert_eq!(d.tracks()[0].codec, codec, "{name} {bit_depth}"),
                None => assert_eq!(d.unmapped_codec_ids(), [name], "{bit_depth}"),
            }
        }
    }

    #[test]
    fn vint_round_trips_through_read_size() {
        for v in [0u64, 1, 100, 127, 128, 16_383, 16_384, 1_000_000] {
            let bytes = vint(v);
            let (got, len, unknown) = read_size(&bytes, 0).expect("decodes");
            assert_eq!(got, v, "value {v}");
            assert_eq!(len, bytes.len());
            assert!(!unknown);
        }
        // All-ones one-byte size is the "unknown size" marker.
        assert_eq!(read_size(&[0xFF], 0), Some((127, 1, true)));
    }

    #[test]
    fn read_id_lengths() {
        assert_eq!(read_id(&[0xA3], 0), Some((0xA3, 1)));
        assert_eq!(read_id(&[0x42, 0x86], 0), Some((0x4286, 2)));
        assert_eq!(
            read_id(&[0x1F, 0x43, 0xB6, 0x75], 0),
            Some((0x1F43_B675, 4))
        );
    }

    #[test]
    fn parses_tracks_and_frames() {
        let mut d = MatroskaDemuxer::new();
        d.push_data(&synthetic_webm());

        assert_eq!(
            d.tracks(),
            &[
                MkvTrack {
                    number: 1,
                    uid: 0,
                    codec: MkvCodec::Vp9,
                    width: 640,
                    height: 480,
                    channels: 0,
                    sample_rate: 0,
                    default_duration_ns: 0,
                    unsupported_encoding: false,
                },
                MkvTrack {
                    number: 2,
                    uid: 0,
                    codec: MkvCodec::Opus,
                    width: 0,
                    height: 0,
                    channels: 2,
                    sample_rate: 48_000,
                    default_duration_ns: 0,
                    unsupported_encoding: false,
                },
            ]
        );

        let frames = d.take_frames();
        assert_eq!(frames.len(), 3, "two video + one audio");
        // Cluster ts 1000 * default scale 1_000_000 ns = 1 ms.
        assert_eq!(
            frames[0],
            MkvFrame {
                track: 1,
                codec: MkvCodec::Vp9,
                pts_ns: 1_000 * 1_000_000,
                duration_ns: 0,
                keyframe: true,
                data: vec![0xDE, 0xAD]
            }
        );
        assert_eq!(frames[1].codec, MkvCodec::Opus);
        assert_eq!(frames[1].data, vec![0xBE, 0xEF]);
        // rel +33 -> (1000+33) * scale.
        assert_eq!(frames[2].pts_ns, 1_033 * 1_000_000);
        assert!(!frames[2].keyframe);
    }

    #[test]
    fn reassembles_across_split_pushes() {
        let webm = synthetic_webm();
        let mut d = MatroskaDemuxer::new();
        // Feed byte by byte: no element completes early, all frames still appear.
        for b in &webm {
            d.push_data(&[*b]);
        }
        assert_eq!(d.tracks().len(), 2);
        assert_eq!(d.take_frames().len(), 3);
    }

    /// An EBML element with an explicit unknown-size marker of `marker_len` bytes
    /// (all-ones), used to build a live-shape Cluster whose end is implicit.
    fn unknown_size_elem(id: &[u8], marker_len: usize, body: &[u8]) -> Vec<u8> {
        let mut out = id.to_vec();
        let mut marker = vec![0xFFu8; marker_len];
        marker[0] = (0xFFu8 >> (marker_len - 1)) | (1 << (8 - marker_len));
        out.extend_from_slice(&marker);
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn demuxes_unknown_size_cluster() {
        // Two live Clusters with unknown size, terminated by each other / EOF.
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &track_entry(1, b"V_VP9", Some((64, 48)), None),
        );
        let cluster0 = unknown_size_elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            1,
            &[
                elem(&[0xE7], &uint_body(0)),
                elem(&[0xA3], &block_body(1, 0, true, &[0xAA])),
                elem(&[0xA3], &block_body(1, 10, false, &[0xBB])),
            ]
            .concat(),
        );
        let cluster1 = unknown_size_elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            1,
            &[
                elem(&[0xE7], &uint_body(100)),
                elem(&[0xA3], &block_body(1, 0, true, &[0xCC])),
            ]
            .concat(),
        );
        let segment = unknown_size_elem(
            &[0x18, 0x53, 0x80, 0x67],
            8,
            &[tracks, cluster0, cluster1].concat(),
        );
        let file = [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat();

        let mut d = MatroskaDemuxer::new();
        d.push_data(&file);
        let frames = d.take_frames();
        assert_eq!(frames.len(), 3, "both live clusters' blocks demux");
        assert_eq!(frames[0].data, vec![0xAA]);
        assert_eq!(frames[0].pts_ns, 0);
        assert_eq!(frames[1].data, vec![0xBB]);
        assert_eq!(frames[1].pts_ns, 10 * 1_000_000);
        assert_eq!(frames[2].data, vec![0xCC]);
        assert_eq!(
            frames[2].pts_ns,
            100 * 1_000_000,
            "second cluster's Timestamp applies"
        );
    }

    #[test]
    fn unknown_size_cluster_skips_benign_children() {
        // A live Cluster may carry CRC-32 / Void children among its blocks; these
        // must be skipped, not treated as the Cluster end (which would drop every
        // following block).
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &track_entry(1, b"V_VP9", Some((64, 48)), None),
        );
        let cluster = unknown_size_elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            1,
            &[
                elem(&[0xBF], &[0x12, 0x34, 0x56, 0x78]), // CRC-32 (benign, often first)
                elem(&[0xE7], &uint_body(0)),             // Timestamp
                elem(&[0xA3], &block_body(1, 0, true, &[0xAA])),
                elem(&[0xEC], &[0x00, 0x00]), // Void (benign padding)
                elem(&[0xA3], &block_body(1, 10, false, &[0xBB])),
            ]
            .concat(),
        );
        let segment = unknown_size_elem(&[0x18, 0x53, 0x80, 0x67], 8, &[tracks, cluster].concat());
        let file = [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat();
        let mut d = MatroskaDemuxer::new();
        d.push_data(&file);
        let frames = d.take_frames();
        assert_eq!(
            frames.len(),
            2,
            "benign children are skipped; both blocks demux"
        );
        assert_eq!(frames[0].data, vec![0xAA]);
        assert_eq!(frames[1].data, vec![0xBB]);
    }

    #[test]
    fn unknown_size_cluster_emits_blocks_incrementally() {
        // A block fully buffered before its Cluster is closed still emits (live
        // playback can't wait for a terminator that may never come).
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &track_entry(1, b"V_VP8", Some((16, 16)), None),
        );
        let mut file = [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[])].concat();
        file.extend_from_slice(&unknown_size_elem(&[0x18, 0x53, 0x80, 0x67], 8, &tracks));
        let mut d = MatroskaDemuxer::new();
        d.push_data(&file);
        // Open a live Cluster header, then feed one Timestamp + one block, no terminator.
        let mut live = unknown_size_elem(&[0x1F, 0x43, 0xB6, 0x75], 1, &[]);
        live.extend_from_slice(&elem(&[0xE7], &uint_body(5)));
        live.extend_from_slice(&elem(&[0xA3], &block_body(1, 0, true, &[0xDD])));
        d.push_data(&live);
        let frames = d.take_frames();
        assert_eq!(
            frames.len(),
            1,
            "the block emits without waiting for a Cluster end"
        );
        assert_eq!(frames[0].pts_ns, 5 * 1_000_000);
    }

    #[test]
    fn truncated_definite_size_cluster_emits_its_complete_blocks() {
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &track_entry(1, b"V_VP8", Some((16, 16)), None),
        );
        let payloads: [&[u8]; 3] = [&[0xAA, 0xAA], &[0xBB, 0xBB], &[0xCC, 0xCC]];
        let mut body = elem(&[0xE7], &uint_body(0));
        for (i, payload) in payloads.iter().enumerate() {
            body.extend_from_slice(&elem(&[0xA3], &block_body(1, i as i16, true, payload)));
        }
        let cluster = elem(&[0x1F, 0x43, 0xB6, 0x75], &body);
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &[tracks, cluster].concat());
        let file = [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat();

        // The file ends one byte into the last block's payload, as a cut-short
        // download does: the Cluster's declared size is never reached.
        let mut d = MatroskaDemuxer::new();
        d.push_data(&file[..file.len() - 1]);
        let frames = d.take_frames();
        let complete = &payloads[..payloads.len() - 1];
        assert_eq!(
            frames.iter().map(|f| f.data.as_slice()).collect::<Vec<_>>(),
            complete.to_vec()
        );
    }

    #[test]
    fn damage_inside_a_cluster_resyncs_at_the_next_cluster() {
        const TRACK: u64 = 1;
        const UNDECLARED_TRACK: u64 = TRACK + 1;
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &track_entry(TRACK, b"V_VP8", Some((16, 16)), None),
        );
        let cluster_id = [0x1F, 0x43, 0xB6, 0x75];
        let block = |payload: &[u8]| elem(&[0xA3], &block_body(TRACK, 0, true, payload));
        let before: &[u8] = &[0xAA];
        let damaged_cluster_good_block: &[u8] = &[0xBB];
        let after: &[u8] = &[0xCC];
        let first = elem(
            &cluster_id,
            &[elem(&[0xE7], &uint_body(0)), block(before)].concat(),
        );
        // A block naming an undeclared track, then an element whose size runs
        // far past the end of the file.
        let mut garbage = elem(&[0xA3], &block_body(UNDECLARED_TRACK, 0, true, &[0xDD]));
        let overrun = vint(1 << 20);
        garbage.extend_from_slice(&[[0xEA].as_slice(), &overrun, &[0xEE; 8]].concat());
        let damaged = elem(
            &cluster_id,
            &[
                elem(&[0xE7], &uint_body(10)),
                block(damaged_cluster_good_block),
                garbage,
            ]
            .concat(),
        );
        let last = elem(
            &cluster_id,
            &[elem(&[0xE7], &uint_body(20)), block(after)].concat(),
        );
        let segment = elem(
            &[0x18, 0x53, 0x80, 0x67],
            &[tracks, first, damaged, last].concat(),
        );
        let file = [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat();
        let expected = vec![before, damaged_cluster_good_block, after];

        let mut whole = MatroskaDemuxer::new();
        whole.push_data(&file);
        let frames = whole.take_frames();
        assert_eq!(
            frames.iter().map(|f| f.data.as_slice()).collect::<Vec<_>>(),
            expected,
            "the blocks either side of the damage demux"
        );

        // The Cluster ID the resync looks for arrives split across pushes.
        let mut bytewise = MatroskaDemuxer::new();
        for b in &file {
            bytewise.push_data(&[*b]);
        }
        let frames = bytewise.take_frames();
        assert_eq!(
            frames.iter().map(|f| f.data.as_slice()).collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn fixed_lacing_block_splits_into_frames() {
        // A SimpleBlock with fixed lacing (flags bit 0x04), two frames, data
        // [0xAA, 0xBB] -> one byte each, both at the cluster timestamp.
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &track_entry(1, b"V_VP8", Some((16, 16)), None),
        );
        let mut laced = vint(1); // track 1
        laced.extend_from_slice(&0i16.to_be_bytes());
        laced.push(0x04); // fixed lacing
        laced.push(0x01); // frame count - 1 = 1 (two frames)
        laced.extend_from_slice(&[0xAA, 0xBB]);
        let cluster = elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[elem(&[0xE7], &uint_body(0)), elem(&[0xA3], &laced)].concat(),
        );
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &[tracks, cluster].concat());
        let file = [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat();

        let mut d = MatroskaDemuxer::new();
        d.push_data(&file);
        let frames = d.take_frames();
        assert_eq!(frames.len(), 2, "fixed lacing splits into two frames");
        assert_eq!(frames[0].data, vec![0xAA]);
        assert_eq!(frames[1].data, vec![0xBB]);
    }

    #[test]
    fn huge_cluster_timestamp_does_not_overflow() {
        // An untrusted Cluster Timestamp >= 2^63 casts to a negative i64; adding a
        // negative block rel must not overflow the abs-timestamp add (a debug
        // panic). It saturates and the existing `< 0` clamp maps it to pts 0.
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &track_entry(1, b"V_VP8", Some((16, 16)), None),
        );
        let cluster = elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[
                elem(&[0xE7], &uint_body(1u64 << 63)),
                elem(&[0xA3], &block_body(1, -1, true, &[0xDD])),
            ]
            .concat(),
        );
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &[tracks, cluster].concat());
        let file = [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat();

        let mut d = MatroskaDemuxer::new();
        d.push_data(&file);
        let frames = d.take_frames();
        assert_eq!(frames.len(), 1, "the block still emits");
        assert_eq!(frames[0].pts_ns, 0, "the overflowing timestamp clamps to 0");
    }

    #[test]
    fn laced_frames_spaced_by_default_duration() {
        // A VP8 track with DefaultDuration 20 ms and a fixed-laced block of two
        // frames: the second advances by one DefaultDuration from the block ts.
        let dur_ns = 20_000_000u64;
        let track_body = [
            elem(&[0xD7], &uint_body(1)),                  // TrackNumber
            elem(&[0x86], b"V_VP8"),                       // CodecID
            elem(&[0x23, 0xE3, 0x83], &uint_body(dur_ns)), // DefaultDuration
            elem(
                &[0xE0],
                &[elem(&[0xB0], &uint_body(16)), elem(&[0xBA], &uint_body(16))].concat(),
            ),
        ]
        .concat();
        let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &elem(&[0xAE], &track_body));

        let mut laced = vint(1); // track 1
        laced.extend_from_slice(&0i16.to_be_bytes());
        laced.push(0x04); // fixed lacing
        laced.push(0x01); // frame count - 1 = 1 (two frames)
        laced.extend_from_slice(&[0xAA, 0xBB]);
        let cluster = elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[elem(&[0xE7], &uint_body(0)), elem(&[0xA3], &laced)].concat(),
        );
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &[tracks, cluster].concat());
        let file = [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat();

        let mut d = MatroskaDemuxer::new();
        d.push_data(&file);
        assert_eq!(d.tracks()[0].default_duration_ns, dur_ns);
        let frames = d.take_frames();
        assert_eq!(frames.len(), 2);
        assert_eq!(
            frames[0].pts_ns, 0,
            "first laced frame at the block timestamp"
        );
        assert_eq!(
            frames[1].pts_ns, dur_ns,
            "second frame advanced by DefaultDuration"
        );
    }

    #[test]
    fn xiph_lacing_splits_with_255_continuation() {
        // Two frames: a 255-byte frame (Xiph size 0xFF 0x00) then a 2-byte one.
        let mut body = vec![1u8]; // frame count - 1 = 1
        body.push(0xFF); // 255...
        body.push(0x00); // ...+ 0 = size 255 for frame 0
        body.extend(vec![0x11u8; 255]);
        body.extend_from_slice(&[0xAB, 0xCD]);
        let frames = split_laced(&body, 1).expect("xiph parses");
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].len(), 255);
        assert_eq!(frames[1], &[0xAB, 0xCD]);
    }

    #[test]
    fn ebml_lacing_splits_with_signed_deltas() {
        // Three frames sized 4, 6, 3. First size = unsigned vint 4 (0x84); the
        // +2 delta is a signed 1-octet vint: unsigned 65 (0x41) -> 0xC1.
        let mut body = vec![2u8]; // frame count - 1 = 2 (three frames)
        body.push(0x84); // first size = 4
        body.push(0xC1); // delta +2 -> second size = 6
        body.extend(vec![0u8; 4 + 6 + 3]); // frame payloads
        let frames = split_laced(&body, 3).expect("ebml parses");
        let lens: Vec<usize> = frames.iter().map(|f| f.len()).collect();
        assert_eq!(lens, vec![4, 6, 3]);
    }

    #[test]
    fn fixed_lacing_rejects_inexact_division() {
        // 5 bytes across two frames doesn't divide evenly: malformed.
        let mut body = vec![1u8]; // two frames
        body.extend_from_slice(&[1, 2, 3, 4, 5]);
        assert!(split_laced(&body, 2).is_none());
    }

    #[test]
    fn mux_demux_round_trip() {
        // Mux a VP9 video track of two frames, then demux the WebM back.
        let spec = MkvTrackSpec {
            codec: MkvCodec::Vp9,
            width: 320,
            height: 240,
            channels: 0,
            sample_rate: 0,
        };
        let mut mux = MatroskaMuxer::new(spec);
        let mut bytes = mux.push_frame(&[1, 2, 3], 0, true);
        bytes.extend_from_slice(&mux.push_frame(&[4, 5], 33_000_000, false));

        let mut d = MatroskaDemuxer::new();
        d.push_data(&bytes);
        assert_eq!(
            d.tracks(),
            &[MkvTrack {
                number: 1,
                uid: 1,
                codec: MkvCodec::Vp9,
                width: 320,
                height: 240,
                channels: 0,
                sample_rate: 0,
                default_duration_ns: 0,
                unsupported_encoding: false,
            }]
        );
        let frames = d.take_frames();
        assert_eq!(frames.len(), 2, "both frames survive the round trip");
        assert_eq!(frames[0].data, vec![1, 2, 3]);
        assert_eq!(frames[0].pts_ns, 0);
        assert!(frames[0].keyframe);
        assert_eq!(frames[1].data, vec![4, 5]);
        assert_eq!(frames[1].pts_ns, 33_000_000); // 33 ms ticks * 1 ms scale
        assert!(!frames[1].keyframe);
    }

    /// A `Tags` element carrying one `Tag` with the given `SimpleTag` pairs and
    /// an empty `Targets` (whole-stream scope).
    fn tags_element(simple: &[(&str, &str)]) -> Vec<u8> {
        let mut tag = elem(&[0x63, 0xC0], &[]); // Targets (empty)
        for (name, value) in simple {
            let body = [
                elem(&[0x45, 0xA3], name.as_bytes()),
                elem(&[0x44, 0x87], value.as_bytes()),
            ]
            .concat();
            tag.extend_from_slice(&elem(&[0x67, 0xC8], &body));
        }
        elem(&[0x12, 0x54, 0xC3, 0x67], &elem(&[0x73, 0x73], &tag))
    }

    #[test]
    fn parses_segment_title_and_tags() {
        let info = elem(&[0x15, 0x49, 0xA9, 0x66], &elem(&[0x7B, 0xA9], b"My Movie")); // Info/Title
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &track_entry(1, b"V_VP9", Some((16, 16)), None),
        );
        let tags = tags_element(&[("ARTIST", "Band"), ("ENCODER", "libvpx")]);
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &[info, tracks, tags].concat());
        let file = [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat();

        let mut d = MatroskaDemuxer::new();
        d.push_data(&file);
        assert_eq!(
            d.tags().tags(),
            &[
                Tag::Title("My Movie".into()),
                Tag::Artist("Band".into()),
                Tag::Encoder("libvpx".into()),
            ]
        );
    }

    #[test]
    fn mux_writes_tags_that_demux_recovers() {
        let spec = MkvTrackSpec {
            codec: MkvCodec::Vp9,
            width: 16,
            height: 16,
            channels: 0,
            sample_rate: 0,
        };
        let tags: TagList = [
            Tag::Title("Clip".into()),
            Tag::Encoder("g2g".into()),
            Tag::Other {
                key: "DIRECTOR".into(),
                value: "Ada".into(),
            },
        ]
        .into_iter()
        .collect();
        let mut mux = MatroskaMuxer::new(spec).with_tags(tags.clone());
        let bytes = mux.push_frame(&[1, 2, 3], 0, true);

        let mut d = MatroskaDemuxer::new();
        d.push_data(&bytes);
        assert_eq!(
            d.tags().tags(),
            tags.tags(),
            "tags survive the mux + demux round trip"
        );
        assert_eq!(
            d.take_frames().len(),
            1,
            "the frame still muxes alongside the tags"
        );
    }

    /// One `Tag` element: a `Targets` naming `track_uid` (0 writes an empty
    /// `Targets`, the whole-stream scope) plus one `SimpleTag` per pair.
    fn scoped_tag(track_uid: u64, simple: &[(&str, &str)]) -> Vec<u8> {
        let targets = if track_uid == 0 {
            Vec::new()
        } else {
            elem(&[0x63, 0xC5], &uint_body(track_uid))
        };
        let mut tag = elem(&[0x63, 0xC0], &targets);
        for (name, value) in simple {
            let body = [
                elem(&[0x45, 0xA3], name.as_bytes()),
                elem(&[0x44, 0x87], value.as_bytes()),
            ]
            .concat();
            tag.extend_from_slice(&elem(&[0x67, 0xC8], &body));
        }
        elem(&[0x73, 0x73], &tag)
    }

    /// A two-track (VP9 + Opus) segment whose `Tags` body is `tags`.
    fn segment_with_tags(tags: &[u8]) -> Vec<u8> {
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &[
                [
                    elem(&[0xD7], &uint_body(1)),
                    elem(&[0x73, 0xC5], &uint_body(11)), // TrackUID
                    elem(&[0x86], b"V_VP9"),
                ]
                .concat(),
                [
                    elem(&[0xD7], &uint_body(2)),
                    elem(&[0x73, 0xC5], &uint_body(22)),
                    elem(&[0x86], b"A_OPUS"),
                ]
                .concat(),
            ]
            .map(|b| elem(&[0xAE], &b))
            .concat(),
        );
        let segment = elem(
            &[0x18, 0x53, 0x80, 0x67],
            &[tracks, elem(&[0x12, 0x54, 0xC3, 0x67], tags)].concat(),
        );
        [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat()
    }

    #[test]
    fn targets_scope_tags_to_their_track() {
        let tags = [
            scoped_tag(0, &[("ARTIST", "Band")]),
            scoped_tag(11, &[("TITLE", "Camera A"), ("LANGUAGE", "eng")]),
            scoped_tag(22, &[("TITLE", "Commentary")]),
        ]
        .concat();
        let mut d = MatroskaDemuxer::new();
        d.push_data(&segment_with_tags(&tags));

        assert_eq!(
            d.tags().tags(),
            &[Tag::Artist("Band".into())],
            "an empty Targets keeps the whole-stream scope"
        );
        assert_eq!(d.tracks()[0].uid, 11);
        assert_eq!(
            d.track_tags(),
            &[
                (
                    11,
                    [Tag::Title("Camera A".into()), Tag::Language("eng".into())]
                        .into_iter()
                        .collect::<TagList>()
                ),
                (
                    22,
                    [Tag::Title("Commentary".into())]
                        .into_iter()
                        .collect::<TagList>()
                ),
            ]
        );
    }

    #[test]
    fn a_tag_targeting_two_tracks_scopes_to_both() {
        let two = elem(
            &[0x73, 0x73],
            &[
                elem(
                    &[0x63, 0xC0],
                    &[
                        elem(&[0x63, 0xC5], &uint_body(11)),
                        elem(&[0x63, 0xC5], &uint_body(22)),
                    ]
                    .concat(),
                ),
                elem(
                    &[0x67, 0xC8],
                    &[elem(&[0x45, 0xA3], b"ARTIST"), elem(&[0x44, 0x87], b"Duo")].concat(),
                ),
            ]
            .concat(),
        );
        let mut d = MatroskaDemuxer::new();
        d.push_data(&segment_with_tags(&two));
        let uids: Vec<u64> = d.track_tags().iter().map(|(uid, _)| *uid).collect();
        assert_eq!(uids, vec![11, 22]);
        assert!(d
            .track_tags()
            .iter()
            .all(|(_, t)| t.tags() == [Tag::Artist("Duo".into())]));
    }

    #[test]
    fn track_uid_zero_is_whole_stream() {
        // TagTrackUID 0 is the spec's "all tracks": not a per-track scope.
        let uid_zero = elem(
            &[0x73, 0x73],
            &[
                elem(&[0x63, 0xC0], &elem(&[0x63, 0xC5], &uint_body(0))),
                elem(
                    &[0x67, 0xC8],
                    &[
                        elem(&[0x45, 0xA3], b"ALBUM"),
                        elem(&[0x44, 0x87], b"Everything"),
                    ]
                    .concat(),
                ),
            ]
            .concat(),
        );
        let mut d = MatroskaDemuxer::new();
        d.push_data(&segment_with_tags(&uid_zero));
        assert_eq!(d.tags().tags(), &[Tag::Album("Everything".into())]);
        assert!(d.track_tags().is_empty());
    }

    #[test]
    fn nested_simple_tags_flatten_to_slash_keys() {
        // A SimpleTag inside a SimpleTag: the child key carries the parent's.
        let inner = elem(
            &[0x67, 0xC8],
            &[
                elem(&[0x45, 0xA3], b"SORT_WITH"),
                elem(&[0x44, 0x87], b"Ada"),
            ]
            .concat(),
        );
        let outer = elem(
            &[0x67, 0xC8],
            &[
                elem(&[0x45, 0xA3], b"ARTIST"),
                elem(&[0x44, 0x87], b"Lovelace"),
                inner,
            ]
            .concat(),
        );
        let tag = elem(
            &[0x73, 0x73],
            &[elem(&[0x63, 0xC0], &[]), outer].concat().to_vec(),
        );
        let mut d = MatroskaDemuxer::new();
        d.push_data(&segment_with_tags(&tag));
        assert_eq!(
            d.tags().tags(),
            &[
                Tag::Artist("Lovelace".into()),
                Tag::Other {
                    key: "ARTIST/SORT_WITH".into(),
                    value: "Ada".into()
                },
            ]
        );
    }

    #[test]
    fn nesting_is_bounded_and_malformed_tags_fail_soft() {
        // Nest deeper than the walk follows: it stops, it does not recurse away.
        let mut nested = elem(
            &[0x67, 0xC8],
            &[elem(&[0x45, 0xA3], b"L9"), elem(&[0x44, 0x87], b"deep")].concat(),
        );
        for _ in 0..64 {
            nested = elem(&[0x67, 0xC8], &[elem(&[0x45, 0xA3], b"N"), nested].concat());
        }
        let mut d = MatroskaDemuxer::new();
        d.push_data(&segment_with_tags(&elem(
            &[0x73, 0x73],
            &[elem(&[0x63, 0xC0], &[]), nested].concat(),
        )));
        assert!(
            d.tags().is_empty(),
            "the deep leaf is past the depth bound, and nothing above it has a value"
        );

        // A Targets whose TagTrackUID body is truncated to nothing, and a
        // SimpleTag with a name but no string: neither panics, neither invents a
        // tag, and the well-formed sibling still parses.
        let odd = [
            elem(
                &[0x73, 0x73],
                &[
                    elem(&[0x63, 0xC0], &elem(&[0x63, 0xC5], &[])),
                    elem(&[0x67, 0xC8], &elem(&[0x45, 0xA3], b"NAMEONLY")),
                ]
                .concat(),
            ),
            scoped_tag(22, &[("TITLE", "Commentary")]),
        ]
        .concat();
        let mut d = MatroskaDemuxer::new();
        d.push_data(&segment_with_tags(&odd));
        assert!(d.tags().is_empty());
        assert_eq!(d.track_tags().len(), 1, "the valid track-scoped Tag parses");
        assert_eq!(d.track_tags()[0].0, 22);

        // A Tags element truncated mid-child yields nothing rather than panicking.
        let truncated = {
            let full = segment_with_tags(&scoped_tag(11, &[("TITLE", "x")]));
            full[..full.len() - 3].to_vec()
        };
        let mut d = MatroskaDemuxer::new();
        d.push_data(&truncated);
        assert!(d.tags().is_empty() && d.track_tags().is_empty());
    }

    #[test]
    fn mux_writes_targets_scoped_tags_that_demux_recovers() {
        let video = MkvTrackConfig {
            spec: MkvTrackSpec {
                codec: MkvCodec::Vp9,
                width: 16,
                height: 16,
                channels: 0,
                sample_rate: 0,
            },
            codec_private: Vec::new(),
        };
        let audio = MkvTrackConfig {
            spec: MkvTrackSpec {
                codec: MkvCodec::Opus,
                width: 0,
                height: 0,
                channels: 2,
                sample_rate: 48_000,
            },
            codec_private: Vec::new(),
        };
        let global: TagList = [Tag::Title("Whole file".into())].into_iter().collect();
        let vid_tags: TagList = [Tag::Artist("Camera A".into())].into_iter().collect();
        let aud_tags: TagList = [
            Tag::Artist("Commentary".into()),
            Tag::Other {
                key: "TAKE".into(),
                value: "2".into(),
            },
        ]
        .into_iter()
        .collect();
        let mut mux = MatroskaMuxer::new_multi(vec![video, audio])
            .with_tags(global.clone())
            .with_track_tags(0, vid_tags.clone())
            .with_track_tags(1, aud_tags.clone())
            // Out of range: no track to scope it to.
            .with_track_tags(9, aud_tags.clone());
        let bytes = mux.push_frame_on(0, &[1, 2, 3], 0, true, 0);

        let mut d = MatroskaDemuxer::new();
        d.push_data(&bytes);
        assert_eq!(
            d.tags().tags(),
            global.tags(),
            "the whole-file Tag survives"
        );
        // The muxer's TrackUIDs are the track numbers, so track 0 is UID 1.
        assert_eq!(
            d.track_tags(),
            &[(1u64, vid_tags), (2u64, aud_tags)],
            "each track's tags come back scoped to its TrackUID"
        );
        assert_eq!(d.tracks()[1].uid, 2, "the Tracks element carries TrackUIDs");
    }

    /// A one-track segment whose `TrackEntry` carries the given extra elements
    /// (`Name` / `Language` / `LanguageBCP47`) after the TrackNumber.
    fn segment_with_track_entry(extra: &[u8]) -> Vec<u8> {
        let mut body = elem(&[0xD7], &uint_body(1));
        body.extend_from_slice(&elem(&[0x73, 0xC5], &uint_body(7)));
        body.extend_from_slice(extra);
        body.extend_from_slice(&elem(&[0x86], b"V_VP9"));
        let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &elem(&[0xAE], &body));
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &tracks);
        [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat()
    }

    #[test]
    fn track_entry_name_and_language_become_per_track_tags() {
        let extra = [
            elem(&[0x53, 0x6E], b"Camera A"),  // Name
            elem(&[0x22, 0xB5, 0x9C], b"eng"), // Language
        ]
        .concat();
        let mut d = MatroskaDemuxer::new();
        d.push_data(&segment_with_track_entry(&extra));
        assert_eq!(
            d.track_entry_tags(),
            &[(
                1u64,
                [Tag::Title("Camera A".into()), Tag::Language("eng".into())]
                    .into_iter()
                    .collect::<TagList>()
            )],
            "keyed by track number, Name first"
        );
        assert!(d.tags().is_empty(), "these are the track's, not the file's");
    }

    #[test]
    fn language_bcp47_wins_over_language() {
        let extra = [
            elem(&[0x22, 0xB5, 0x9C], b"fre"),
            elem(&[0x22, 0xB5, 0x9D], b"fr-CA"),
        ]
        .concat();
        let mut d = MatroskaDemuxer::new();
        d.push_data(&segment_with_track_entry(&extra));
        assert_eq!(
            d.track_entry_tags()[0].1.tags(),
            &[Tag::Language("fr-CA".into())],
            "the BCP-47 form is the more precise one"
        );
    }

    #[test]
    fn absent_language_yields_no_tag_and_bad_strings_fail_soft() {
        // No Name / Language at all: the spec's implicit "eng" default is not
        // metadata the file stated, so nothing is surfaced.
        let mut d = MatroskaDemuxer::new();
        d.push_data(&segment_with_track_entry(&[]));
        assert!(d.track_entry_tags().is_empty());
        assert_eq!(d.tracks().len(), 1, "the track itself still parses");

        // Non-UTF-8 Name and an oversized Language: both skipped, the track parses.
        let oversized = vec![b'x'; MAX_STRING_ELEMENT_LEN + 1];
        let extra = [
            elem(&[0x53, 0x6E], &[0xFF, 0xFE]),
            elem(&[0x22, 0xB5, 0x9C], &oversized),
        ]
        .concat();
        let mut d = MatroskaDemuxer::new();
        d.push_data(&segment_with_track_entry(&extra));
        assert!(d.track_entry_tags().is_empty());
        assert_eq!(d.tracks()[0].number, 1);
    }

    #[test]
    fn mux_writes_language_and_name_in_the_track_entry_not_the_tags() {
        let spec = MkvTrackSpec {
            codec: MkvCodec::Vp9,
            width: 16,
            height: 16,
            channels: 0,
            sample_rate: 0,
        };
        let track: TagList = [
            Tag::Title("Camera A".into()),
            Tag::Language("eng".into()),
            Tag::Artist("Ada".into()),
        ]
        .into_iter()
        .collect();
        let mut mux = MatroskaMuxer::new(spec).with_track_tags(0, track);
        let bytes = mux.push_frame(&[1, 2, 3], 0, true);

        let mut d = MatroskaDemuxer::new();
        d.push_data(&bytes);
        assert_eq!(
            d.track_entry_tags(),
            &[(
                1u64,
                [Tag::Title("Camera A".into()), Tag::Language("eng".into())]
                    .into_iter()
                    .collect::<TagList>()
            )],
            "title and language ride the TrackEntry"
        );
        assert_eq!(
            d.track_tags(),
            &[(1u64, [Tag::Artist("Ada".into())].into_iter().collect())],
            "everything else stays a Targets-scoped Tag"
        );
        // No double-write: the Tags element names ARTIST and nothing else.
        assert!(!contains(&bytes, b"LANGUAGE") && !contains(&bytes, b"TITLE"));
        assert!(contains(&bytes, b"ARTIST"));
    }

    /// A 20 ms stereo Opus packet: the TOC byte alone fixes the length, which is
    /// all the trim arithmetic reads.
    fn opus_packet(payload: u8) -> Vec<u8> {
        vec![0xFC, payload, payload]
    }

    fn opus_track() -> MkvTrackConfig {
        MkvTrackConfig {
            spec: MkvTrackSpec {
                codec: MkvCodec::Opus,
                width: 0,
                height: 0,
                channels: 2,
                sample_rate: 48_000,
            },
            codec_private: crate::opusparse::synth_opus_head(2, 48_000),
        }
    }

    #[test]
    fn opus_track_entry_carries_codec_delay_and_seek_pre_roll() {
        let mut mux = MatroskaMuxer::new_multi(vec![opus_track()]);
        let bytes = mux.push_frame_on(0, &opus_packet(1), 0, true, 0);
        let mut d = MatroskaDemuxer::new();
        d.push_data(&bytes);

        // The synthesized header's 312-sample lookahead, in ns.
        let head = d
            .codec_private(1)
            .expect("the Opus track has a CodecPrivate");
        assert_eq!(
            opus_codec_delay_ns(head),
            6_500_000,
            "312 samples at 48 kHz is 6.5 ms"
        );
        // Written on the wire, as the elements a player reads.
        assert!(contains(&bytes, &[0x56, 0xAA, 0x83, 0x63, 0x2E, 0xA0]));
        assert!(contains(
            &bytes,
            &[0x56, 0xBB, 0x84, 0x04, 0xC4, 0xB4, 0x00]
        ));
    }

    #[test]
    fn a_trimmed_opus_packet_writes_a_block_group_with_its_discard() {
        let mut mux = MatroskaMuxer::new_multi(vec![opus_track()]);
        // A full packet stays a SimpleBlock; a short final one becomes a
        // BlockGroup carrying the trim.
        let mut bytes = mux.push_frame_on(0, &opus_packet(1), 0, true, 20_000_000);
        assert!(
            !contains(&bytes, &[0xA0]) || !contains(&bytes, &[0x75, 0xA2]),
            "an untrimmed packet needs no BlockGroup"
        );
        bytes.extend_from_slice(&mux.push_frame_on(
            0,
            &opus_packet(2),
            20_000_000,
            true,
            6_500_000,
        ));

        let mut d = MatroskaDemuxer::new();
        d.push_data(&bytes);
        let frames = d.take_frames();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].duration_ns, 0, "a SimpleBlock declares nothing");
        assert_eq!(
            frames[1].duration_ns, 6_500_000,
            "the trim survives the round trip in ns, not rounded to the 1 ms grid"
        );
        // DiscardPadding is the ns spelling: 20 ms packet - 6.5 ms kept.
        assert!(contains(
            &bytes,
            &[0x75, 0xA2, 0x84, 0x00, 0xCD, 0xFE, 0x60]
        ));
    }

    #[test]
    fn discard_padding_beats_block_duration_and_bad_values_fail_soft() {
        // A BlockGroup naming both: the ns element wins over the ms one.
        let group = |extra: Vec<u8>| {
            let mut g = elem(&[0xA1], &block_body(1, 0, true, &opus_packet(3)));
            g.extend_from_slice(&extra);
            elem(&[0xA0], &g)
        };
        let both = [
            elem(&[0x9B], &uint_body(7)),                   // BlockDuration 7 ms
            elem(&[0x75, 0xA2], &[0x00, 0xCD, 0xFE, 0x60]), // DiscardPadding 13.5 ms
        ]
        .concat();
        let mut d = MatroskaDemuxer::new();
        d.push_data(&opus_segment(&group(both)));
        assert_eq!(
            d.take_frames()[0].duration_ns,
            6_500_000,
            "20 ms packet less a 13.5 ms discard, not the rounded 7 ms"
        );

        // A negative discard (the spec's leading-padding form), one longer than
        // the packet, and an oversized body: none of them invent a duration, and
        // the BlockDuration underneath still stands.
        for bad in [
            alloc::vec![0xFF, 0xFF],             // -1 ns
            alloc::vec![0x7F, 0xFF, 0xFF, 0xFF], // 2.1 s > packet
            alloc::vec![0u8; 9],                 // wider than i64
        ] {
            let extra = [elem(&[0x9B], &uint_body(20)), elem(&[0x75, 0xA2], &bad)].concat();
            let mut d = MatroskaDemuxer::new();
            d.push_data(&opus_segment(&group(extra)));
            let frames = d.take_frames();
            assert_eq!(frames.len(), 1, "the block still parses: {bad:?}");
            assert_eq!(
                frames[0].duration_ns, 20_000_000,
                "the BlockDuration stands, the bogus discard is ignored: {bad:?}"
            );
        }
    }

    /// A one-Opus-track segment whose single Cluster holds `block`.
    fn opus_segment(block: &[u8]) -> Vec<u8> {
        let track = [
            elem(&[0xD7], &uint_body(1)),
            elem(&[0x86], b"A_OPUS"),
            elem(
                &[0xE1],
                &[
                    elem(&[0x9F], &uint_body(2)),
                    elem(&[0xB5], &(48_000f32).to_be_bytes()),
                ]
                .concat(),
            ),
        ]
        .concat();
        let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &elem(&[0xAE], &track));
        let cluster = elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[elem(&[0xE7], &uint_body(0)), block.to_vec()].concat(),
        );
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &[tracks, cluster].concat());
        [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat()
    }

    #[test]
    fn the_two_pass_duration_is_the_highest_block_end() {
        let spec = MkvTrackSpec {
            codec: MkvCodec::Vp9,
            width: 16,
            height: 16,
            channels: 0,
            sample_rate: 0,
        };
        let mut mux = MatroskaMuxer::new(spec).with_two_pass();
        let mut file = mux.push_frame(&[1], 0, true);
        file.extend_from_slice(&mux.push_frame(&[2], 40_000_000, false));
        // The last frame declares no duration, so it lasts the previous gap:
        // 80 ms + 40 ms.
        file.extend_from_slice(&mux.push_frame(&[3], 80_000_000, false));
        finalize_seekable(&mux, &mut file);

        let (off, value) = mux
            .duration_patch()
            .expect("the two-pass mode reserves one");
        assert_eq!(f64::from_be_bytes(value), 120.0, "in TimestampScale ticks");
        assert_eq!(
            &file[off..off + 8],
            &value,
            "and the placeholder in Info was patched with it"
        );
    }

    #[test]
    fn a_declared_duration_beats_the_frame_gap_and_streaming_reserves_nothing() {
        let mut mux = MatroskaMuxer::new_multi(vec![opus_track()]);
        // Streaming: no placeholder to patch, whatever the frames say.
        mux.push_frame_on(0, &opus_packet(1), 0, true, 20_000_000);
        assert!(mux.duration_patch().is_none());

        // Two-pass: the trimmed final packet's own 6.5 ms rounds to 7 ticks, so
        // the file ends at 20 + 7, not at a whole packet past the last block.
        let mut mux = MatroskaMuxer::new_multi(vec![opus_track()]).with_two_pass();
        mux.push_frame_on(0, &opus_packet(1), 0, true, 20_000_000);
        mux.push_frame_on(0, &opus_packet(2), 20_000_000, true, 6_500_000);
        let (_, value) = mux.duration_patch().expect("reserved");
        assert_eq!(f64::from_be_bytes(value), 27.0);
    }

    #[test]
    fn signed_element_bodies_round_trip() {
        for v in [
            0i64,
            1,
            -1,
            127,
            128,
            -128,
            -129,
            13_500_000,
            i64::MAX,
            i64::MIN,
        ] {
            assert_eq!(read_int(&int_bytes(v)), v, "{v}");
        }
        assert_eq!(int_bytes(13_500_000), alloc::vec![0x00, 0xCD, 0xFE, 0x60]);
        assert_eq!(read_int(&[]), 0, "an empty body is no value");
        assert_eq!(read_int(&[0u8; 9]), 0, "wider than i64 is no value");
    }

    #[test]
    fn mux_writes_no_tags_element_when_only_track_entry_tags_are_set() {
        let spec = MkvTrackSpec {
            codec: MkvCodec::Vp9,
            width: 16,
            height: 16,
            channels: 0,
            sample_rate: 0,
        };
        let track: TagList = [Tag::Language("deu".into())].into_iter().collect();
        let bytes = MatroskaMuxer::new(spec)
            .with_track_tags(0, track)
            .push_frame(&[0], 0, true);
        // 0x1254C367 is the Tags element id: nothing left for it to carry.
        assert!(!contains(&bytes, &[0x12, 0x54, 0xC3, 0x67]));
        let mut d = MatroskaDemuxer::new();
        d.push_data(&bytes);
        assert_eq!(
            d.track_entry_tags()[0].1.tags(),
            &[Tag::Language("deu".into())]
        );
    }

    fn contains(haystack: &[u8], needle: &[u8]) -> bool {
        haystack.windows(needle.len()).any(|w| w == needle)
    }

    #[test]
    fn mux_without_tags_writes_no_tags_element() {
        let spec = MkvTrackSpec {
            codec: MkvCodec::Vp9,
            width: 16,
            height: 16,
            channels: 0,
            sample_rate: 0,
        };
        let bytes = MatroskaMuxer::new(spec).push_frame(&[0], 0, true);
        let mut d = MatroskaDemuxer::new();
        d.push_data(&bytes);
        assert!(d.tags().is_empty());
    }

    fn count_clusters(bytes: &[u8]) -> usize {
        bytes
            .windows(4)
            .filter(|w| *w == [0x1F, 0x43, 0xB6, 0x75])
            .count()
    }

    #[test]
    fn batches_frames_within_span_into_one_cluster() {
        let spec = MkvTrackSpec {
            codec: MkvCodec::Vp9,
            width: 16,
            height: 16,
            channels: 0,
            sample_rate: 0,
        };
        let mut mux = MatroskaMuxer::new(spec); // default 1000 ms span
        let mut out = mux.push_frame(&[1], 0, true);
        out.extend_from_slice(&mux.push_frame(&[2], 100_000_000, false)); // 100 ms
        out.extend_from_slice(&mux.push_frame(&[3], 200_000_000, false)); // 200 ms

        assert_eq!(
            count_clusters(&out),
            1,
            "frames within the span share one Cluster"
        );
        let mut d = MatroskaDemuxer::new();
        d.push_data(&out);
        let frames = d.take_frames();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].pts_ns, 0);
        assert_eq!(frames[1].pts_ns, 100_000_000);
        assert_eq!(
            frames[2].pts_ns, 200_000_000,
            "rel timestamps within the batched Cluster"
        );
    }

    #[test]
    fn opens_a_new_cluster_past_the_span() {
        let spec = MkvTrackSpec {
            codec: MkvCodec::Vp9,
            width: 16,
            height: 16,
            channels: 0,
            sample_rate: 0,
        };
        let mut mux = MatroskaMuxer::new(spec).with_max_cluster_span_ms(500);
        let mut out = mux.push_frame(&[1], 0, true);
        out.extend_from_slice(&mux.push_frame(&[2], 600_000_000, true)); // 600 ms > 500 ms

        assert_eq!(
            count_clusters(&out),
            2,
            "a frame past the span opens a new Cluster"
        );
        let mut d = MatroskaDemuxer::new();
        d.push_data(&out);
        let frames = d.take_frames();
        assert_eq!(frames.len(), 2);
        assert_eq!(
            frames[1].pts_ns, 600_000_000,
            "second Cluster carries its own base"
        );
    }

    /// M828: a live muxer keeps no Cues state. The bytes it writes are the same
    /// as a recording muxer's, so only the index the caller would append at EOS
    /// differs, and there is nothing to append.
    #[test]
    fn without_cues_writes_the_same_bytes_and_no_index() {
        let spec = MkvTrackSpec {
            codec: MkvCodec::Vp9,
            width: 16,
            height: 16,
            channels: 0,
            sample_rate: 0,
        };
        let push = |mux: &mut MatroskaMuxer| {
            let mut out = mux.push_frame(&[1], 0, true);
            out.extend_from_slice(&mux.push_frame(&[2], 600_000_000, true));
            out
        };
        let mut recording = MatroskaMuxer::new(spec).with_max_cluster_span_ms(500);
        let mut live = MatroskaMuxer::new(spec)
            .with_max_cluster_span_ms(500)
            .without_cues();
        assert_eq!(push(&mut recording), push(&mut live), "identical output");
        assert!(!recording.finish().is_empty(), "the recording indexes both");
        assert!(
            live.finish().is_empty(),
            "the live muxer collected no cue points to index"
        );
    }

    #[test]
    fn mux_writes_cues_that_demux_resolves_to_the_clusters() {
        // Two keyframes a span apart open two Clusters; the EOS Cues index should
        // point each CueTime at its Cluster's byte offset (M375, the write side of
        // M373's read).
        let spec = MkvTrackSpec {
            codec: MkvCodec::Vp9,
            width: 16,
            height: 16,
            channels: 0,
            sample_rate: 0,
        };
        let mut mux = MatroskaMuxer::new(spec).with_max_cluster_span_ms(500);
        let mut out = mux.push_frame(&[1], 0, true);
        out.extend_from_slice(&mux.push_frame(&[2], 200_000_000, false)); // same Cluster, non-key
        out.extend_from_slice(&mux.push_frame(&[3], 600_000_000, true)); // 600 ms > 500 ms: Cluster 2
        let cues = mux.finish();
        assert!(!cues.is_empty(), "a Cues element is produced at finish");
        out.extend_from_slice(&cues);

        let mut d = MatroskaDemuxer::new();
        d.push_data(&out);
        // One CuePoint per Cluster holding a keyframe (the non-key frame did not add
        // a second cue to Cluster 1).
        assert_eq!(
            d.cues().len(),
            2,
            "one cue per keyframe-bearing Cluster, deduped"
        );
        assert_eq!(d.cues()[0].time_ns, 0);
        assert_eq!(d.cues()[1].time_ns, 600_000_000);

        // Each cue's resolved absolute offset lands exactly on a Cluster element id.
        for target in [0u64, 600_000_000] {
            let off = d.cue_seek_offset(target).expect("offset for a cued time") as usize;
            assert_eq!(
                &out[off..off + 4],
                &[0x1F, 0x43, 0xB6, 0x75],
                "cue points at a Cluster"
            );
        }
    }

    #[test]
    fn mux_writes_webm_doctype_for_vp9() {
        let spec = MkvTrackSpec {
            codec: MkvCodec::Vp9,
            width: 16,
            height: 16,
            channels: 0,
            sample_rate: 0,
        };
        let bytes = MatroskaMuxer::new(spec).push_frame(&[0], 0, true);
        // The DocType string appears in the EBML header for a WebM codec.
        assert!(bytes.windows(4).any(|w| w == b"webm"), "VP9 muxes as WebM");
    }

    /// A `CuePoint`: CueTime + CueTrackPositions(CueTrack, CueClusterPosition).
    fn cue_point(time: u64, track: u64, pos: u64) -> Vec<u8> {
        let tp = [
            elem(&[0xF7], &uint_body(track)),
            elem(&[0xF1], &uint_body(pos)),
        ]
        .concat();
        let body = [elem(&[0xB3], &uint_body(time)), elem(&[0xB7], &tp)].concat();
        elem(&[0xBB], &body)
    }

    #[test]
    fn parses_cues_and_resolves_seek_offsets() {
        let ebml = elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]);
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &track_entry(1, b"V_VP9", Some((320, 240)), None),
        );
        let cluster = |ts: u64, frame: &[u8]| {
            elem(
                &[0x1F, 0x43, 0xB6, 0x75],
                &[
                    elem(&[0xE7], &uint_body(ts)),
                    elem(&[0xA3], &block_body(1, 0, true, frame)),
                ]
                .concat(),
            )
        };
        let cluster0 = cluster(0, &[0xAA]);
        let cluster1 = cluster(1000, &[0xBB]); // 1000 ms with the default 1 ms scale = 1 s
                                               // Cues sit after the Clusters (the common layout): positions are relative
                                               // to the Segment data start.
        let cluster0_pos = tracks.len() as u64;
        let cluster1_pos = (tracks.len() + cluster0.len()) as u64;
        let cues = elem(
            &[0x1C, 0x53, 0xBB, 0x6B],
            &[
                cue_point(0, 1, cluster0_pos),
                cue_point(1000, 1, cluster1_pos),
            ]
            .concat(),
        );
        let body = [tracks.clone(), cluster0.clone(), cluster1, cues].concat();
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &body);
        let seg_data_pos = ebml.len() as u64 + (segment.len() - body.len()) as u64;
        let file = [ebml, segment].concat();

        let mut d = MatroskaDemuxer::new();
        d.push_data(&file);

        assert_eq!(d.cues().len(), 2);
        assert_eq!(
            d.cues()[0],
            CuePoint {
                time_ns: 0,
                cluster_position: cluster0_pos
            }
        );
        assert_eq!(
            d.cues()[1],
            CuePoint {
                time_ns: 1000 * DEFAULT_TIMESTAMP_SCALE,
                cluster_position: cluster1_pos
            }
        );

        // A seek at the second cue's exact time lands on Cluster1; the absolute
        // offset points at the Cluster element id in the file.
        let off = d.cue_seek_offset(1000 * DEFAULT_TIMESTAMP_SCALE).unwrap();
        assert_eq!(off, seg_data_pos + cluster1_pos);
        assert_eq!(
            &file[off as usize..off as usize + 4],
            &[0x1F, 0x43, 0xB6, 0x75]
        );

        // A target between cues snaps back to the largest cue at/before it (Cluster0).
        assert_eq!(
            d.cue_seek_offset(500 * DEFAULT_TIMESTAMP_SCALE),
            Some(seg_data_pos + cluster0_pos)
        );
        // A target before the first cue clamps to it.
        assert_eq!(d.cue_seek_offset(0), Some(seg_data_pos + cluster0_pos));
    }

    #[test]
    fn cue_seek_offset_none_without_cues() {
        let mut d = MatroskaDemuxer::new();
        d.push_data(&synthetic_webm()); // no Cues element
        assert!(d.cues().is_empty());
        assert_eq!(
            d.cue_seek_offset(0),
            None,
            "no index -> caller re-scans from 0"
        );
    }

    /// A SeekHead Seek entry: SeekID (the target element ID, whole) + SeekPosition.
    fn seek_entry(target_id: &[u8], pos: u64) -> Vec<u8> {
        let body = [
            elem(&[0x53, 0xAB], target_id),
            elem(&[0x53, 0xAC], &uint_body(pos)),
        ]
        .concat();
        elem(&[0x4D, 0xBB], &body)
    }

    #[test]
    fn seekhead_locates_the_cues_element() {
        let ebml = elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]);
        // SeekHead points at Cues at position 5000 (relative to Segment data).
        let seekhead = elem(
            &[0x11, 0x4D, 0x9B, 0x74],
            &[
                seek_entry(&[0x16, 0x54, 0xAE, 0x6B], 100), // Tracks (ignored)
                seek_entry(&[0x1C, 0x53, 0xBB, 0x6B], 5000), // Cues
            ]
            .concat(),
        );
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &track_entry(1, b"V_VP9", Some((320, 240)), None),
        );
        let body = [seekhead, tracks].concat();
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &body);
        let seg_data_pos = ebml.len() as u64 + (segment.len() - body.len()) as u64;
        let file = [ebml, segment].concat();

        let mut d = MatroskaDemuxer::new();
        d.push_data(&file);
        // The Cues are not parsed (none in the file), but their location is known.
        assert!(d.cues().is_empty());
        assert_eq!(d.cue_index_offset(), Some(seg_data_pos + 5000));
    }

    #[test]
    fn cue_index_offset_none_without_seekhead() {
        let mut d = MatroskaDemuxer::new();
        d.push_data(&synthetic_webm());
        assert_eq!(d.cue_index_offset(), None);
    }
}
