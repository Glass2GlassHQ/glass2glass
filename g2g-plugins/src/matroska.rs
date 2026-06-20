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
//! Scope (v1): a single Segment. Both definite-size and unknown-size Clusters are
//! handled, the latter (the live-streaming shape) descended into and its children
//! parsed until the next top-level element ends it. SimpleBlock / Block frames,
//! including all three lacing modes (Xiph / EBML / fixed), are split; laced frames
//! share the block timestamp. Cues (seeking), BlockGroup reference tracking, and
//! per-frame timestamp interpolation from DefaultDuration are follow-ups.

use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::{Tag, TagList};

/// EBML / Matroska element IDs (kept whole, length marker included). The demuxer
/// skips the EBML header by its size and ignores TrackType (the CodecID pins the
/// media type), but the muxer writes both, so they are named here too.
const ID_EBML: u32 = 0x1A45_DFA3;
const ID_SEGMENT: u32 = 0x1853_8067;
const ID_INFO: u32 = 0x1549_A966;
const ID_TIMESTAMP_SCALE: u32 = 0x002A_D7B1;
const ID_TITLE: u32 = 0x7BA9;
const ID_TAGS: u32 = 0x1254_C367;
const ID_TAG: u32 = 0x7373;
const ID_TARGETS: u32 = 0x63C0;
const ID_SIMPLE_TAG: u32 = 0x67C8;
const ID_TAG_NAME: u32 = 0x45A3;
const ID_TAG_STRING: u32 = 0x4487;
const ID_TRACKS: u32 = 0x1654_AE6B;
const ID_TRACK_ENTRY: u32 = 0x00AE;
const ID_TRACK_NUMBER: u32 = 0x00D7;
const ID_TRACK_TYPE: u32 = 0x0083;
const ID_CODEC_ID: u32 = 0x0086;
const ID_DEFAULT_DURATION: u32 = 0x0023_E383;
const ID_VIDEO: u32 = 0x00E0;
const ID_PIXEL_WIDTH: u32 = 0x00B0;
const ID_PIXEL_HEIGHT: u32 = 0x00BA;
const ID_AUDIO: u32 = 0x00E1;
const ID_CHANNELS: u32 = 0x009F;
const ID_SAMPLING_FREQ: u32 = 0x00B5;
const ID_CLUSTER: u32 = 0x1F43_B675;
const ID_TIMESTAMP: u32 = 0x00E7;
const ID_SIMPLE_BLOCK: u32 = 0x00A3;
const ID_BLOCK_GROUP: u32 = 0x00A0;
const ID_BLOCK: u32 = 0x00A1;

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
    Aac,
    Opus,
    /// A `CodecID` this demuxer does not map to a g2g caps type.
    Other,
}

impl MkvCodec {
    /// Map a Matroska `CodecID` string to a codec. AAC has profile suffixes
    /// (`A_AAC/MPEG4/LC`, ...), so it is matched by prefix; the rest are exact.
    fn from_codec_id(id: &[u8]) -> MkvCodec {
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
        } else if id.starts_with(b"A_AAC") {
            MkvCodec::Aac
        } else if id == b"A_OPUS" {
            MkvCodec::Opus
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
            MkvCodec::Aac => b"A_AAC",
            MkvCodec::Opus => b"A_OPUS",
            MkvCodec::Other => return None,
        })
    }

    /// True for the WebM codec subset, so the muxer can write the `webm` DocType.
    pub fn is_webm(self) -> bool {
        matches!(self, MkvCodec::Vp8 | MkvCodec::Vp9 | MkvCodec::Av1 | MkvCodec::Opus)
    }

    /// `1` for video, `2` for audio (the Matroska `TrackType`).
    fn track_type(self) -> u8 {
        match self {
            MkvCodec::Aac | MkvCodec::Opus => 2,
            _ => 1,
        }
    }
}

/// One elementary stream announced by a `TrackEntry`. Geometry (`width` /
/// `height`) is set for video, `channels` / `sample_rate` for audio; the others
/// stay zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MkvTrack {
    pub number: u64,
    pub codec: MkvCodec,
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub sample_rate: u32,
    /// Nanoseconds per frame from `DefaultDuration` (0 when the track omits it).
    /// Spaces the frames of a laced block; an unscaled value, unlike block
    /// timestamps.
    pub default_duration_ns: u64,
}

/// One demuxed frame (a SimpleBlock / Block payload) of an elementary stream.
#[derive(Debug, Clone, PartialEq)]
pub struct MkvFrame {
    pub track: u64,
    pub codec: MkvCodec,
    /// Presentation timestamp in nanoseconds (cluster + block, scaled).
    pub pts_ns: u64,
    pub keyframe: bool,
    pub data: Vec<u8>,
}

/// Incremental Matroska demuxer: feed bytes, drain [`MkvFrame`]s.
#[derive(Debug)]
pub struct MatroskaDemuxer {
    buf: Vec<u8>,
    in_segment: bool,
    timestamp_scale: u64,
    tracks: Vec<MkvTrack>,
    tags: TagList,
    /// The current Timestamp of an open unknown-size Cluster (the live shape).
    /// `Some` while its children are being parsed at the top level, `None`
    /// otherwise. A definite-size Cluster never sets this (it is consumed whole).
    open_cluster_ts: Option<u64>,
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
            buf: Vec::new(),
            in_segment: false,
            timestamp_scale: DEFAULT_TIMESTAMP_SCALE,
            tracks: Vec::new(),
            tags: TagList::new(),
            open_cluster_ts: None,
            completed: Vec::new(),
        }
    }

    /// The elementary streams announced by `Tracks` (empty until it is seen).
    pub fn tracks(&self) -> &[MkvTrack] {
        &self.tracks
    }

    /// The stream metadata from the Segment's `Tags` element and the `Info`
    /// `Title` (empty until either is seen). Accumulates across pushes.
    pub fn tags(&self) -> &TagList {
        &self.tags
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
    /// an unknown-size Cluster are descended into (their children are read at this
    /// level); every other element is consumed once its definite-size body is
    /// fully buffered.
    fn drain_elements(&mut self) {
        loop {
            let Some((id, id_len)) = read_id(&self.buf, 0) else { return };
            let Some((size, size_len, unknown)) = read_size(&self.buf, id_len) else { return };
            let header = id_len + size_len;

            if id == ID_SEGMENT {
                // Descend: a Segment's children are parsed at this level, so its
                // own size (definite or unknown) is never needed.
                self.buf.drain(..header);
                self.in_segment = true;
                continue;
            }

            // An unknown-size Cluster (the live shape) is likewise descended into:
            // its children are parsed at this level until the next top-level
            // element ends it.
            if id == ID_CLUSTER && unknown {
                self.buf.drain(..header);
                self.open_cluster_ts = Some(0);
                continue;
            }

            // Inside an open unknown-size Cluster, its own children are decoded
            // here; any other element closes the Cluster and is handled normally.
            if self.open_cluster_ts.is_some() {
                match id {
                    ID_TIMESTAMP | ID_SIMPLE_BLOCK | ID_BLOCK_GROUP => {
                        if unknown {
                            return; // a Cluster child must carry a definite size
                        }
                        let Some(total) = header.checked_add(size as usize) else { return };
                        if self.buf.len() < total {
                            return;
                        }
                        if id == ID_TIMESTAMP {
                            self.open_cluster_ts = Some(read_uint(&self.buf[header..total]));
                        } else {
                            let ts = self.open_cluster_ts.unwrap_or(0);
                            let frames = parse_block_element(
                                id,
                                &self.buf[header..total],
                                ts,
                                self.timestamp_scale,
                                &self.tracks,
                            );
                            self.completed.extend(frames);
                        }
                        self.buf.drain(..total);
                        continue;
                    }
                    _ => self.open_cluster_ts = None, // end the Cluster, handle id below
                }
            }

            // Every other element is consumed whole; a definite size tells us where
            // it ends (an unknown size here is a container we do not descend).
            if unknown {
                return;
            }
            let Some(total) = header.checked_add(size as usize) else { return };
            if self.buf.len() < total {
                return; // wait for the rest of this element
            }

            if self.in_segment {
                match id {
                    ID_INFO => {
                        if let Some(scale) = parse_timestamp_scale(&self.buf[header..total]) {
                            self.timestamp_scale = scale;
                        }
                        if let Some(title) = parse_info_title(&self.buf[header..total]) {
                            self.tags.push(Tag::Title(title));
                        }
                    }
                    ID_TRACKS => self.tracks = parse_tracks(&self.buf[header..total]),
                    ID_TAGS => {
                        for tag in parse_tags(&self.buf[header..total]) {
                            self.tags.push(tag);
                        }
                    }
                    ID_CLUSTER => {
                        let frames =
                            parse_cluster(&self.buf[header..total], &self.tracks, self.timestamp_scale);
                        self.completed.extend(frames);
                    }
                    _ => {} // SeekHead / Cues / Chapters / Void, etc.
                }
            }
            // (elements before the Segment, e.g. the EBML header, are skipped.)
            self.buf.drain(..total);
        }
    }
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
    children(info).find(|(id, _)| *id == ID_TIMESTAMP_SCALE).map(|(_, d)| read_uint(d))
}

/// The Segment `Info` `Title` (the whole-file title), if present and valid UTF-8.
fn parse_info_title(info: &[u8]) -> Option<String> {
    let (_, data) = children(info).find(|(id, _)| *id == ID_TITLE)?;
    core::str::from_utf8(data).ok().map(String::from)
}

/// Parse the Segment `Tags` element into a flat list of [`Tag`]s. Each `Tag`'s
/// `SimpleTag` children carry a `TagName` / `TagString` pair; the conventional
/// uppercase Matroska names (`TITLE`, `ARTIST`, ...) map through
/// [`Tag::from_key_value`]. Targets and nested SimpleTags are ignored (v1: the
/// whole-stream tags, no per-track scoping).
fn parse_tags(body: &[u8]) -> Vec<Tag> {
    let mut out = Vec::new();
    for (tag_id, tag) in children(body) {
        if tag_id != ID_TAG {
            continue;
        }
        for (sid, simple) in children(tag) {
            if sid == ID_SIMPLE_TAG {
                if let Some(t) = parse_simple_tag(simple) {
                    out.push(t);
                }
            }
        }
    }
    out
}

/// One `SimpleTag`: a `TagName` keyed `TagString` value (both UTF-8). A
/// `TagBinary` value or a missing name/string yields nothing.
fn parse_simple_tag(body: &[u8]) -> Option<Tag> {
    let mut name: Option<&str> = None;
    let mut value: Option<&str> = None;
    for (id, data) in children(body) {
        match id {
            ID_TAG_NAME => name = core::str::from_utf8(data).ok(),
            ID_TAG_STRING => value = core::str::from_utf8(data).ok(),
            _ => {}
        }
    }
    Some(Tag::from_key_value(name?, value?))
}

fn parse_tracks(body: &[u8]) -> Vec<MkvTrack> {
    let mut tracks = Vec::new();
    for (id, entry) in children(body) {
        if id == ID_TRACK_ENTRY {
            if let Some(t) = parse_track_entry(entry) {
                tracks.push(t);
            }
        }
    }
    tracks
}

fn parse_track_entry(body: &[u8]) -> Option<MkvTrack> {
    let mut number = 0u64;
    let mut codec_id: &[u8] = &[];
    let mut width = 0u32;
    let mut height = 0u32;
    let mut channels = 0u8;
    let mut sample_rate = 0u32;
    let mut default_duration_ns = 0u64;
    for (id, data) in children(body) {
        match id {
            ID_TRACK_NUMBER => number = read_uint(data),
            ID_CODEC_ID => codec_id = data,
            ID_DEFAULT_DURATION => default_duration_ns = read_uint(data),
            ID_VIDEO => {
                for (vid, vdata) in children(data) {
                    match vid {
                        ID_PIXEL_WIDTH => width = read_uint(vdata) as u32,
                        ID_PIXEL_HEIGHT => height = read_uint(vdata) as u32,
                        _ => {}
                    }
                }
            }
            ID_AUDIO => {
                for (aid, adata) in children(data) {
                    match aid {
                        ID_CHANNELS => channels = read_uint(adata) as u8,
                        ID_SAMPLING_FREQ => sample_rate = read_float(adata) as u32,
                        _ => {}
                    }
                }
            }
            _ => {} // TrackType is implied by the CodecID prefix; FlagLacing etc. ignored
        }
    }
    if number == 0 {
        return None;
    }
    Some(MkvTrack {
        number,
        codec: MkvCodec::from_codec_id(codec_id),
        width,
        height,
        channels,
        sample_rate,
        default_duration_ns,
    })
}

/// Parse one Cluster's body, appending its frames. The Cluster `Timestamp`
/// precedes its blocks (spec-mandated), so it is set before any block is decoded.
fn parse_cluster(body: &[u8], tracks: &[MkvTrack], scale: u64) -> Vec<MkvFrame> {
    let mut cluster_ts = 0u64;
    let mut frames = Vec::new();
    for (id, data) in children(body) {
        match id {
            ID_TIMESTAMP => cluster_ts = read_uint(data),
            ID_SIMPLE_BLOCK => parse_block(data, cluster_ts, scale, tracks, &mut frames),
            ID_BLOCK_GROUP => {
                for (bid, bdata) in children(data) {
                    if bid == ID_BLOCK {
                        parse_block(bdata, cluster_ts, scale, tracks, &mut frames);
                    }
                }
            }
            _ => {}
        }
    }
    frames
}

/// Parse a single Cluster child block element (a `SimpleBlock` or `BlockGroup`)
/// into frames, for the unknown-size-Cluster path where children are decoded one
/// at a time. The Cluster `Timestamp` is handled by the caller.
fn parse_block_element(
    id: u32,
    body: &[u8],
    cluster_ts: u64,
    scale: u64,
    tracks: &[MkvTrack],
) -> Vec<MkvFrame> {
    let mut frames = Vec::new();
    match id {
        ID_SIMPLE_BLOCK => parse_block(body, cluster_ts, scale, tracks, &mut frames),
        ID_BLOCK_GROUP => {
            for (bid, bdata) in children(body) {
                if bid == ID_BLOCK {
                    parse_block(bdata, cluster_ts, scale, tracks, &mut frames);
                }
            }
        }
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
    out: &mut Vec<MkvFrame>,
) {
    let Some((track, tn_len, _)) = read_size(block, 0) else { return };
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
    let abs = cluster_ts as i64 + rel as i64;
    let pts_ns = if abs < 0 { 0 } else { (abs as u64).saturating_mul(scale) };
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
        let frame_pts = pts_ns.saturating_add(i as u64 * default_duration_ns);
        out.push(MkvFrame { track, codec, pts_ns: frame_pts, keyframe, data: data.to_vec() });
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
    if count == 0 || data.is_empty() || data.len() % count != 0 {
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
            cur += raw as i64 - bias;
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

/// The single track a [`MatroskaMuxer`] writes.
const MUX_TRACK_NUMBER: u64 = 1;

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

/// Matroska / WebM multiplexer for a single track (M115): writes the EBML header,
/// an unknown-size Segment, Info + Tracks, then one Cluster per frame. The
/// inverse of [`MatroskaDemuxer`]; the [`crate::mkvmux::MkvMux`] element wraps it.
///
/// Scope (v1): one track, one frame per Cluster (correct but unbatched), default
/// TimestampScale (1 ms). Cues, multi-track, and Cluster batching are follow-ups.
#[derive(Debug)]
pub struct MatroskaMuxer {
    spec: MkvTrackSpec,
    tags: TagList,
    header_written: bool,
}

impl MatroskaMuxer {
    pub fn new(spec: MkvTrackSpec) -> Self {
        Self { spec, tags: TagList::new(), header_written: false }
    }

    /// Attach stream metadata, written as a `Tags` element after Tracks on the
    /// first frame (the inverse of [`MatroskaDemuxer::tags`]).
    pub fn with_tags(mut self, tags: TagList) -> Self {
        self.tags = tags;
        self
    }

    /// Mux one frame, writing the EBML header + Segment + Info + Tracks (+ Tags
    /// when present) on the first call, then a Cluster (Timestamp + SimpleBlock)
    /// for this frame.
    pub fn push_frame(&mut self, data: &[u8], pts_ns: u64, keyframe: bool) -> Vec<u8> {
        let mut out = Vec::new();
        if !self.header_written {
            let doctype: &[u8] = if self.spec.codec.is_webm() { b"webm" } else { b"matroska" };
            out.extend_from_slice(&ebml_header(doctype));
            // Segment with unknown size: its children run to end of stream.
            id_bytes(ID_SEGMENT, &mut out);
            out.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
            out.extend_from_slice(&info_element());
            out.extend_from_slice(&tracks_element(&self.spec));
            if !self.tags.is_empty() {
                out.extend_from_slice(&tags_element(&self.tags));
            }
            self.header_written = true;
        }
        let ts = pts_ns / DEFAULT_TIMESTAMP_SCALE;
        let block = build_simple_block(MUX_TRACK_NUMBER, 0, keyframe, data);
        let mut cluster = elem_vec(ID_TIMESTAMP, &uint_bytes(ts));
        cluster.extend_from_slice(&elem_vec(ID_SIMPLE_BLOCK, &block));
        out.extend_from_slice(&elem_vec(ID_CLUSTER, &cluster));
        out
    }
}

/// A minimal but valid EBML header naming the DocType (`matroska` / `webm`).
fn ebml_header(doctype: &[u8]) -> Vec<u8> {
    let mut h = elem_vec(0x4286, &[1]); // EBMLVersion
    h.extend_from_slice(&elem_vec(0x42F7, &[1])); // EBMLReadVersion
    h.extend_from_slice(&elem_vec(0x42F2, &[4])); // EBMLMaxIDLength
    h.extend_from_slice(&elem_vec(0x42F3, &[8])); // EBMLMaxSizeLength
    h.extend_from_slice(&elem_vec(0x4282, doctype)); // DocType
    h.extend_from_slice(&elem_vec(0x4287, &[2])); // DocTypeVersion
    h.extend_from_slice(&elem_vec(0x4285, &[2])); // DocTypeReadVersion
    elem_vec(ID_EBML, &h)
}

fn info_element() -> Vec<u8> {
    elem_vec(ID_INFO, &elem_vec(ID_TIMESTAMP_SCALE, &uint_bytes(DEFAULT_TIMESTAMP_SCALE)))
}

fn tracks_element(spec: &MkvTrackSpec) -> Vec<u8> {
    let codec_id = spec.codec.codec_id().unwrap_or(b"");
    let mut entry = elem_vec(ID_TRACK_NUMBER, &uint_bytes(MUX_TRACK_NUMBER));
    entry.extend_from_slice(&elem_vec(ID_TRACK_TYPE, &uint_bytes(spec.codec.track_type() as u64)));
    entry.extend_from_slice(&elem_vec(ID_CODEC_ID, codec_id));
    if spec.codec.track_type() == 1 {
        let mut v = elem_vec(ID_PIXEL_WIDTH, &uint_bytes(spec.width as u64));
        v.extend_from_slice(&elem_vec(ID_PIXEL_HEIGHT, &uint_bytes(spec.height as u64)));
        entry.extend_from_slice(&elem_vec(ID_VIDEO, &v));
    } else {
        let mut a = elem_vec(ID_CHANNELS, &uint_bytes(spec.channels.max(1) as u64));
        a.extend_from_slice(&elem_vec(ID_SAMPLING_FREQ, &(spec.sample_rate as f64).to_be_bytes()));
        entry.extend_from_slice(&elem_vec(ID_AUDIO, &a));
    }
    let entry = elem_vec(ID_TRACK_ENTRY, &entry);
    elem_vec(ID_TRACKS, &entry)
}

/// A whole-stream `Tags` element: one `Tag` with an empty `Targets` and a
/// `SimpleTag` (TagName + TagString) per entry. The inverse of [`parse_tags`];
/// the typed keys write their conventional uppercase Matroska names.
fn tags_element(tags: &TagList) -> Vec<u8> {
    let mut tag = elem_vec(ID_TARGETS, &[]);
    for t in tags.tags() {
        let (name, value) = tag_name_value(t);
        let mut simple = elem_vec(ID_TAG_NAME, name.as_bytes());
        simple.extend_from_slice(&elem_vec(ID_TAG_STRING, value.as_bytes()));
        tag.extend_from_slice(&elem_vec(ID_SIMPLE_TAG, &simple));
    }
    elem_vec(ID_TAGS, &elem_vec(ID_TAG, &tag))
}

/// A tag's Matroska `TagName` / `TagString` pair. Typed keys use the conventional
/// uppercase names so they round-trip back to the same variant through
/// [`Tag::from_key_value`]; [`Tag::Other`] keeps its stored key.
fn tag_name_value(tag: &Tag) -> (&str, &str) {
    match tag {
        Tag::Title(v) => ("TITLE", v),
        Tag::Artist(v) => ("ARTIST", v),
        Tag::Album(v) => ("ALBUM", v),
        Tag::Encoder(v) => ("ENCODER", v),
        Tag::Language(v) => ("LANGUAGE", v),
        Tag::Comment(v) => ("COMMENT", v),
        Tag::Other { key, value } => (key, value),
    }
}

/// A SimpleBlock body: track-number VINT, signed relative timestamp, flags, data.
fn build_simple_block(track: u64, rel: i16, keyframe: bool, data: &[u8]) -> Vec<u8> {
    let mut b = encode_vint(track);
    b.extend_from_slice(&rel.to_be_bytes());
    b.push(if keyframe { 0x80 } else { 0x00 }); // keyframe flag, no lacing
    b.extend_from_slice(data);
    b
}

/// One EBML element: serialized id, a size VINT, then the body.
fn elem_vec(id: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    id_bytes(id, &mut out);
    out.extend_from_slice(&encode_vint(body.len() as u64));
    out.extend_from_slice(body);
    out
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

    fn track_entry(number: u64, codec: &[u8], video: Option<(u32, u32)>, audio: Option<(u8, u32)>) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&elem(&[0xD7], &uint_body(number)));
        body.extend_from_slice(&elem(&[0x86], codec));
        if let Some((w, h)) = video {
            let v = [elem(&[0xB0], &uint_body(w as u64)), elem(&[0xBA], &uint_body(h as u64))].concat();
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
        assert_eq!(read_id(&[0x1F, 0x43, 0xB6, 0x75], 0), Some((0x1F43_B675, 4)));
    }

    #[test]
    fn parses_tracks_and_frames() {
        let mut d = MatroskaDemuxer::new();
        d.push_data(&synthetic_webm());

        assert_eq!(
            d.tracks(),
            &[
                MkvTrack { number: 1, codec: MkvCodec::Vp9, width: 640, height: 480, channels: 0, sample_rate: 0, default_duration_ns: 0 },
                MkvTrack { number: 2, codec: MkvCodec::Opus, width: 0, height: 0, channels: 2, sample_rate: 48_000, default_duration_ns: 0 },
            ]
        );

        let frames = d.take_frames();
        assert_eq!(frames.len(), 3, "two video + one audio");
        // Cluster ts 1000 * default scale 1_000_000 ns = 1 ms.
        assert_eq!(frames[0], MkvFrame { track: 1, codec: MkvCodec::Vp9, pts_ns: 1_000 * 1_000_000, keyframe: true, data: vec![0xDE, 0xAD] });
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
        let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &track_entry(1, b"V_VP9", Some((64, 48)), None));
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
            &[elem(&[0xE7], &uint_body(100)), elem(&[0xA3], &block_body(1, 0, true, &[0xCC]))]
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
        assert_eq!(frames[2].pts_ns, 100 * 1_000_000, "second cluster's Timestamp applies");
    }

    #[test]
    fn unknown_size_cluster_emits_blocks_incrementally() {
        // A block fully buffered before its Cluster is closed still emits (live
        // playback can't wait for a terminator that may never come).
        let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &track_entry(1, b"V_VP8", Some((16, 16)), None));
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
        assert_eq!(frames.len(), 1, "the block emits without waiting for a Cluster end");
        assert_eq!(frames[0].pts_ns, 5 * 1_000_000);
    }

    #[test]
    fn fixed_lacing_block_splits_into_frames() {
        // A SimpleBlock with fixed lacing (flags bit 0x04), two frames, data
        // [0xAA, 0xBB] -> one byte each, both at the cluster timestamp.
        let tracks =
            elem(&[0x16, 0x54, 0xAE, 0x6B], &track_entry(1, b"V_VP8", Some((16, 16)), None));
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
    fn laced_frames_spaced_by_default_duration() {
        // A VP8 track with DefaultDuration 20 ms and a fixed-laced block of two
        // frames: the second advances by one DefaultDuration from the block ts.
        let dur_ns = 20_000_000u64;
        let track_body = [
            elem(&[0xD7], &uint_body(1)),                  // TrackNumber
            elem(&[0x86], b"V_VP8"),                       // CodecID
            elem(&[0x23, 0xE3, 0x83], &uint_body(dur_ns)), // DefaultDuration
            elem(&[0xE0], &[elem(&[0xB0], &uint_body(16)), elem(&[0xBA], &uint_body(16))].concat()),
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
        assert_eq!(frames[0].pts_ns, 0, "first laced frame at the block timestamp");
        assert_eq!(frames[1].pts_ns, dur_ns, "second frame advanced by DefaultDuration");
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
        let spec =
            MkvTrackSpec { codec: MkvCodec::Vp9, width: 320, height: 240, channels: 0, sample_rate: 0 };
        let mut mux = MatroskaMuxer::new(spec);
        let mut bytes = mux.push_frame(&[1, 2, 3], 0, true);
        bytes.extend_from_slice(&mux.push_frame(&[4, 5], 33_000_000, false));

        let mut d = MatroskaDemuxer::new();
        d.push_data(&bytes);
        assert_eq!(
            d.tracks(),
            &[MkvTrack { number: 1, codec: MkvCodec::Vp9, width: 320, height: 240, channels: 0, sample_rate: 0, default_duration_ns: 0 }]
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
            let body = [elem(&[0x45, 0xA3], name.as_bytes()), elem(&[0x44, 0x87], value.as_bytes())]
                .concat();
            tag.extend_from_slice(&elem(&[0x67, 0xC8], &body));
        }
        elem(&[0x12, 0x54, 0xC3, 0x67], &elem(&[0x73, 0x73], &tag))
    }

    #[test]
    fn parses_segment_title_and_tags() {
        let info = elem(&[0x15, 0x49, 0xA9, 0x66], &elem(&[0x7B, 0xA9], b"My Movie")); // Info/Title
        let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &track_entry(1, b"V_VP9", Some((16, 16)), None));
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
        let spec =
            MkvTrackSpec { codec: MkvCodec::Vp9, width: 16, height: 16, channels: 0, sample_rate: 0 };
        let tags: TagList = [
            Tag::Title("Clip".into()),
            Tag::Encoder("g2g".into()),
            Tag::Other { key: "DIRECTOR".into(), value: "Ada".into() },
        ]
        .into_iter()
        .collect();
        let mut mux = MatroskaMuxer::new(spec).with_tags(tags.clone());
        let bytes = mux.push_frame(&[1, 2, 3], 0, true);

        let mut d = MatroskaDemuxer::new();
        d.push_data(&bytes);
        assert_eq!(d.tags().tags(), tags.tags(), "tags survive the mux + demux round trip");
        assert_eq!(d.take_frames().len(), 1, "the frame still muxes alongside the tags");
    }

    #[test]
    fn mux_without_tags_writes_no_tags_element() {
        let spec =
            MkvTrackSpec { codec: MkvCodec::Vp9, width: 16, height: 16, channels: 0, sample_rate: 0 };
        let bytes = MatroskaMuxer::new(spec).push_frame(&[0], 0, true);
        let mut d = MatroskaDemuxer::new();
        d.push_data(&bytes);
        assert!(d.tags().is_empty());
    }

    #[test]
    fn mux_writes_webm_doctype_for_vp9() {
        let spec =
            MkvTrackSpec { codec: MkvCodec::Vp9, width: 16, height: 16, channels: 0, sample_rate: 0 };
        let bytes = MatroskaMuxer::new(spec).push_frame(&[0], 0, true);
        // The DocType string appears in the EBML header for a WebM codec.
        assert!(bytes.windows(4).any(|w| w == b"webm"), "VP9 muxes as WebM");
    }
}
