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
//! Scope (v1): a single Segment; definite-size elements (an unknown-size Segment
//! is fine, the demuxer never needs its size, but unknown-size Clusters, the live
//! streaming shape, are not handled); no-lacing SimpleBlock / Block frames (laced
//! blocks are counted and skipped). Cues (seeking), BlockGroup reference
//! tracking, and lacing are follow-ups.

use alloc::vec::Vec;

/// EBML / Matroska element IDs (kept whole, length marker included). The EBML
/// header (`0x1A45DFA3`) and any other pre-Segment element are skipped by their
/// size, so only the elements parsed below are named. TrackType is unused: the
/// CodecID string already pins the media type.
const ID_SEGMENT: u32 = 0x1853_8067;
const ID_INFO: u32 = 0x1549_A966;
const ID_TIMESTAMP_SCALE: u32 = 0x002A_D7B1;
const ID_TRACKS: u32 = 0x1654_AE6B;
const ID_TRACK_ENTRY: u32 = 0x00AE;
const ID_TRACK_NUMBER: u32 = 0x00D7;
const ID_CODEC_ID: u32 = 0x0086;
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
    completed: Vec<MkvFrame>,
    laced_skipped: u64,
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
            completed: Vec::new(),
            laced_skipped: 0,
        }
    }

    /// The elementary streams announced by `Tracks` (empty until it is seen).
    pub fn tracks(&self) -> &[MkvTrack] {
        &self.tracks
    }

    /// Drain the frames demuxed so far.
    pub fn take_frames(&mut self) -> Vec<MkvFrame> {
        core::mem::take(&mut self.completed)
    }

    /// Count of laced blocks skipped (v1 does not split lacing).
    pub fn laced_blocks_skipped(&self) -> u64 {
        self.laced_skipped
    }

    /// Feed container bytes. Complete top-level elements are parsed as they
    /// arrive; a partial trailing element waits for the next call.
    pub fn push_data(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
        self.drain_elements();
    }

    /// Consume whole top-level elements from the front of `buf`. The Segment is
    /// descended into (its children are read at this level); every other element
    /// is consumed once its definite-size body is fully buffered.
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

            // Every other element is consumed whole; v1 needs a definite size to
            // know where it ends (unknown-size Clusters are the live shape).
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
                    }
                    ID_TRACKS => self.tracks = parse_tracks(&self.buf[header..total]),
                    ID_CLUSTER => {
                        let (frames, laced) = parse_cluster(
                            &self.buf[header..total],
                            &self.tracks,
                            self.timestamp_scale,
                        );
                        self.laced_skipped += laced;
                        self.completed.extend(frames);
                    }
                    _ => {} // SeekHead / Cues / Tags / Chapters / Void, etc.
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
    for (id, data) in children(body) {
        match id {
            ID_TRACK_NUMBER => number = read_uint(data),
            ID_CODEC_ID => codec_id = data,
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
    })
}

/// Parse one Cluster's body into frames, returning `(frames, laced_skipped)`.
/// The Cluster `Timestamp` precedes its blocks (spec-mandated), so it is set
/// before any block is decoded.
fn parse_cluster(body: &[u8], tracks: &[MkvTrack], scale: u64) -> (Vec<MkvFrame>, u64) {
    let mut cluster_ts = 0u64;
    let mut frames = Vec::new();
    let mut laced = 0u64;
    for (id, data) in children(body) {
        match id {
            ID_TIMESTAMP => cluster_ts = read_uint(data),
            ID_SIMPLE_BLOCK => match parse_block(data, cluster_ts, scale, tracks) {
                BlockResult::Frame(f) => frames.push(f),
                BlockResult::Laced => laced += 1,
                BlockResult::Drop => {}
            },
            ID_BLOCK_GROUP => {
                for (bid, bdata) in children(data) {
                    if bid == ID_BLOCK {
                        match parse_block(bdata, cluster_ts, scale, tracks) {
                            BlockResult::Frame(f) => frames.push(f),
                            BlockResult::Laced => laced += 1,
                            BlockResult::Drop => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
    (frames, laced)
}

enum BlockResult {
    Frame(MkvFrame),
    Laced,
    Drop,
}

/// Parse a (Simple)Block: track-number VINT, 2-byte signed relative timestamp,
/// a flags byte, then the frame. Laced blocks are reported, not split (v1).
fn parse_block(block: &[u8], cluster_ts: u64, scale: u64, tracks: &[MkvTrack]) -> BlockResult {
    let Some((track, tn_len, _)) = read_size(block, 0) else { return BlockResult::Drop };
    let mut pos = tn_len;
    if pos + 3 > block.len() {
        return BlockResult::Drop;
    }
    let rel = i16::from_be_bytes([block[pos], block[pos + 1]]);
    pos += 2;
    let flags = block[pos];
    pos += 1;
    if (flags >> 1) & 0x03 != 0 {
        return BlockResult::Laced; // Xiph / EBML / fixed lacing: not split in v1
    }
    let Some(codec) = tracks.iter().find(|t| t.number == track).map(|t| t.codec) else {
        return BlockResult::Drop;
    };
    let abs = cluster_ts as i64 + rel as i64;
    let pts_ns = if abs < 0 { 0 } else { (abs as u64).saturating_mul(scale) };
    BlockResult::Frame(MkvFrame {
        track,
        codec,
        pts_ns,
        keyframe: flags & 0x80 != 0,
        data: block[pos..].to_vec(),
    })
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
                MkvTrack { number: 1, codec: MkvCodec::Vp9, width: 640, height: 480, channels: 0, sample_rate: 0 },
                MkvTrack { number: 2, codec: MkvCodec::Opus, width: 0, height: 0, channels: 2, sample_rate: 48_000 },
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

    #[test]
    fn laced_block_is_counted_not_emitted() {
        // A SimpleBlock with fixed lacing (flags 0x04) is skipped in v1.
        let tracks = elem(
            &[0x16, 0x54, 0xAE, 0x6B],
            &track_entry(1, b"V_VP8", Some((16, 16)), None),
        );
        let mut laced = vint(1); // track 1
        laced.extend_from_slice(&0i16.to_be_bytes());
        laced.push(0x04); // fixed lacing
        laced.push(0x01); // (lace frame count - 1)
        laced.extend_from_slice(&[0xAA, 0xBB]);
        let cluster = elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[elem(&[0xE7], &uint_body(0)), elem(&[0xA3], &laced)].concat(),
        );
        let segment = elem(&[0x18, 0x53, 0x80, 0x67], &[tracks, cluster].concat());
        let file = [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat();

        let mut d = MatroskaDemuxer::new();
        d.push_data(&file);
        assert!(d.take_frames().is_empty(), "laced block not emitted");
        assert_eq!(d.laced_blocks_skipped(), 1);
    }
}
