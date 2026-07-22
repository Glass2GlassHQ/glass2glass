//! Wire serialization of a [`PipelinePacket`] (M551, the distributed-graph
//! primitive).
//!
//! A hand-rolled, versioned, little-endian binary codec that turns any
//! [`PipelinePacket`] into a self-contained byte buffer and back. This is the
//! target-agnostic core of the "remote" transport pair (`RemoteSink` /
//! `RemoteSrc` in `g2g-plugins`): serialize a packet here, ship the bytes over
//! any byte transport (TCP, WebSocket, ...), and reconstruct the identical
//! packet on the far side. Cutting an edge in a graph and re-linking the two
//! halves across a network boundary is then just a `RemoteSink` on the near
//! side and a `RemoteSrc` on the far side, with the whole `PipelinePacket`
//! stream (leading `CapsChanged`, `Segment`, `DataFrame`s, mid-stream caps
//! refinement, `Flush`, `Eos`) flowing over the wire.
//!
//! `no_std + alloc`, no external dependency: the codec is pure computation
//! (bytes in, bytes out), so it compiles on every target the core does,
//! including `wasm32` (a browser client can speak the same wire format as a
//! native peer, generalizing the bespoke M549 detect-server shim into a first
//! class primitive).
//!
//! # What crosses the boundary
//!
//! Only CPU memory serializes. [`MemoryDomain::System`] frames carry their bytes
//! verbatim; [`MemoryDomain::SystemView`] frames are materialized to dense
//! row-major bytes (the one copy a strided chain pays when it must leave the
//! process). A device-resident domain (CUDA, D3D11, wgpu, DMABUF, ...) is a
//! bare pointer into another process's GPU and cannot be shipped, so
//! [`encode_packet`] returns [`WireError::UnsupportedDomain`]: put an explicit
//! download element (e.g. `CudaDownload`) before a `RemoteSink` to reach the
//! wire, exactly as the pipeline already requires to reach a CPU sink.
//!
//! Per-frame metadata (the `metadata` feature) is carried when both peers build
//! with it: the two concrete meta types, `AnalyticsMeta` (the detection graph)
//! and `BlobMeta` (opaque tagged side-data), round-trip in band, so a detection
//! computed on one machine arrives attached to its frame on another. Metadata
//! is the last field of a `DataFrame` body, so a `metadata`-off receiver simply
//! ignores a `metadata`-on sender's meta payload rather than mis-parsing the
//! stream (a mixed-feature deployment degrades to no metadata, never to
//! corruption).

use alloc::string::String;
use alloc::vec::Vec;

use crate::caps::{
    AudioFormat, ByteStreamEncoding, Caps, Dim, Rate, RawVideoFormat, TensorDType, TensorLayout,
    TensorShape, TextFormat, VideoCodec,
};
use crate::frame::{Frame, FrameTiming, PipelinePacket};
use crate::memory::{MemoryDomain, SystemSlice};
use crate::meta::FrameMetaSet;
use crate::segment::Segment;
use crate::tensor::MAX_TENSOR_RANK;

/// Wire format version, the first byte of every encoded packet. Bumped on any
/// incompatible layout change so a decoder rejects a mismatched peer up front
/// rather than mis-parsing.
pub const WIRE_VERSION: u8 = 1;

/// Failure decoding (or encoding) a [`PipelinePacket`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireError {
    /// The buffer ended mid-field (a truncated or corrupt message).
    Truncated,
    /// An unknown version byte, packet tag, enum discriminant, or invalid UTF-8
    /// in a string field.
    BadTag,
    /// A device-resident / foreign memory domain that cannot be serialized over
    /// a byte transport (only [`MemoryDomain::System`] / `SystemView` can).
    UnsupportedDomain,
}

// ---- packet / domain / meta tags ----

const PKT_CAPS_CHANGED: u8 = 0;
const PKT_DATA_FRAME: u8 = 1;
const PKT_EOS: u8 = 2;
const PKT_FLUSH: u8 = 3;
const PKT_SEGMENT: u8 = 4;

const DOMAIN_SYSTEM: u8 = 0;

#[cfg_attr(not(feature = "metadata"), allow(dead_code))]
const META_ANALYTICS: u8 = 0;
#[cfg_attr(not(feature = "metadata"), allow(dead_code))]
const META_BLOB: u8 = 1;

// ---- primitive writer ----

struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Writer { buf: Vec::new() }
    }
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn bool(&mut self, v: bool) {
        self.u8(v as u8);
    }
    /// Only the metadata path (`AnalyticsMeta` boxes / confidences) writes f32s.
    #[cfg_attr(not(feature = "metadata"), allow(dead_code))]
    fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }
    fn f64(&mut self, v: f64) {
        self.u64(v.to_bits());
    }
    /// A length-prefixed byte slice (`u32` length then the bytes).
    fn bytes(&mut self, b: &[u8]) {
        self.u32(b.len() as u32);
        self.buf.extend_from_slice(b);
    }
    /// Only the metadata path (`BlobMeta` headers) writes strings.
    #[cfg_attr(not(feature = "metadata"), allow(dead_code))]
    fn str(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }
}

// ---- primitive reader ----

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], WireError> {
        let end = self.pos.checked_add(n).ok_or(WireError::Truncated)?;
        let slice = self.buf.get(self.pos..end).ok_or(WireError::Truncated)?;
        self.pos = end;
        Ok(slice)
    }
    fn u8(&mut self) -> Result<u8, WireError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, WireError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, WireError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn bool(&mut self) -> Result<bool, WireError> {
        Ok(self.u8()? != 0)
    }
    /// Only the metadata path reads f32s.
    #[cfg_attr(not(feature = "metadata"), allow(dead_code))]
    fn f32(&mut self) -> Result<f32, WireError> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn f64(&mut self) -> Result<f64, WireError> {
        Ok(f64::from_bits(self.u64()?))
    }
    fn bytes(&mut self) -> Result<Vec<u8>, WireError> {
        let len = self.u32()? as usize;
        Ok(self.take(len)?.to_vec())
    }
    /// Only the metadata path reads strings.
    #[cfg_attr(not(feature = "metadata"), allow(dead_code))]
    fn str(&mut self) -> Result<String, WireError> {
        String::from_utf8(self.bytes()?).map_err(|_| WireError::BadTag)
    }
}

// ---- enum <-> u8 (exhaustive matches so a new variant is a compile error here) ----

fn video_codec_to_u8(c: VideoCodec) -> u8 {
    match c {
        VideoCodec::H264 => 0,
        VideoCodec::H265 => 1,
        VideoCodec::Av1 => 2,
        VideoCodec::Vp8 => 3,
        VideoCodec::Vp9 => 4,
        VideoCodec::Mjpeg => 5,
        VideoCodec::Mpeg4Part2 => 6,
        VideoCodec::JpegXs => 7,
    }
}
fn video_codec_from_u8(v: u8) -> Result<VideoCodec, WireError> {
    Ok(match v {
        0 => VideoCodec::H264,
        1 => VideoCodec::H265,
        2 => VideoCodec::Av1,
        3 => VideoCodec::Vp8,
        4 => VideoCodec::Vp9,
        5 => VideoCodec::Mjpeg,
        6 => VideoCodec::Mpeg4Part2,
        7 => VideoCodec::JpegXs,
        _ => return Err(WireError::BadTag),
    })
}

/// Map a [`RawVideoFormat`] to its stable wire byte. Public so out-of-crate
/// transports (e.g. the local DMABUF socket) reuse the one canonical numbering
/// instead of duplicating it.
pub fn raw_format_to_u8(f: RawVideoFormat) -> u8 {
    match f {
        RawVideoFormat::Nv12 => 0,
        RawVideoFormat::I420 => 1,
        RawVideoFormat::Rgba8 => 2,
        RawVideoFormat::Bgra8 => 3,
        RawVideoFormat::Yuyv => 4,
        RawVideoFormat::I420p10 => 5,
        RawVideoFormat::I420p12 => 6,
        RawVideoFormat::I422 => 7,
        RawVideoFormat::I422p10 => 8,
        RawVideoFormat::I422p12 => 9,
        RawVideoFormat::I444 => 10,
        RawVideoFormat::I444p10 => 11,
        RawVideoFormat::I444p12 => 12,
    }
}
/// Inverse of [`raw_format_to_u8`]; errors on an unknown byte (never trust the
/// transport).
pub fn raw_format_from_u8(v: u8) -> Result<RawVideoFormat, WireError> {
    Ok(match v {
        0 => RawVideoFormat::Nv12,
        1 => RawVideoFormat::I420,
        2 => RawVideoFormat::Rgba8,
        3 => RawVideoFormat::Bgra8,
        4 => RawVideoFormat::Yuyv,
        5 => RawVideoFormat::I420p10,
        6 => RawVideoFormat::I420p12,
        7 => RawVideoFormat::I422,
        8 => RawVideoFormat::I422p10,
        9 => RawVideoFormat::I422p12,
        10 => RawVideoFormat::I444,
        11 => RawVideoFormat::I444p10,
        12 => RawVideoFormat::I444p12,
        _ => return Err(WireError::BadTag),
    })
}

fn audio_format_to_u8(f: AudioFormat) -> u8 {
    match f {
        AudioFormat::Aac => 0,
        AudioFormat::Opus => 1,
        AudioFormat::PcmS16Le => 2,
        AudioFormat::PcmF32Le => 3,
        AudioFormat::PcmS24Le => 4,
        AudioFormat::Mulaw => 5,
        AudioFormat::Alaw => 6,
        AudioFormat::ImaAdpcm => 7,
        AudioFormat::Mp2 => 8,
        AudioFormat::Ac3 => 9,
        AudioFormat::Flac => 10,
    }
}
fn audio_format_from_u8(v: u8) -> Result<AudioFormat, WireError> {
    Ok(match v {
        0 => AudioFormat::Aac,
        1 => AudioFormat::Opus,
        2 => AudioFormat::PcmS16Le,
        3 => AudioFormat::PcmF32Le,
        4 => AudioFormat::PcmS24Le,
        5 => AudioFormat::Mulaw,
        6 => AudioFormat::Alaw,
        7 => AudioFormat::ImaAdpcm,
        8 => AudioFormat::Mp2,
        9 => AudioFormat::Ac3,
        10 => AudioFormat::Flac,
        _ => return Err(WireError::BadTag),
    })
}

fn bytestream_to_u8(e: ByteStreamEncoding) -> u8 {
    match e {
        ByteStreamEncoding::MpegTs => 0,
        ByteStreamEncoding::Matroska => 1,
        ByteStreamEncoding::Ogg => 2,
        ByteStreamEncoding::Flv => 3,
        ByteStreamEncoding::IsoBmff => 4,
        ByteStreamEncoding::Mp4 => 5,
        ByteStreamEncoding::Ivf => 6,
    }
}
fn bytestream_from_u8(v: u8) -> Result<ByteStreamEncoding, WireError> {
    Ok(match v {
        0 => ByteStreamEncoding::MpegTs,
        1 => ByteStreamEncoding::Matroska,
        2 => ByteStreamEncoding::Ogg,
        3 => ByteStreamEncoding::Flv,
        4 => ByteStreamEncoding::IsoBmff,
        5 => ByteStreamEncoding::Mp4,
        6 => ByteStreamEncoding::Ivf,
        _ => return Err(WireError::BadTag),
    })
}

fn text_format_to_u8(f: TextFormat) -> u8 {
    match f {
        TextFormat::Utf8 => 0,
        TextFormat::PangoMarkup => 1,
        TextFormat::Srt => 2,
        TextFormat::WebVtt => 3,
        TextFormat::Ssa => 4,
        TextFormat::Ttml => 5,
    }
}
fn text_format_from_u8(v: u8) -> Result<TextFormat, WireError> {
    Ok(match v {
        0 => TextFormat::Utf8,
        1 => TextFormat::PangoMarkup,
        2 => TextFormat::Srt,
        3 => TextFormat::WebVtt,
        4 => TextFormat::Ssa,
        5 => TextFormat::Ttml,
        _ => return Err(WireError::BadTag),
    })
}

fn dtype_to_u8(d: TensorDType) -> u8 {
    match d {
        TensorDType::F16 => 0,
        TensorDType::F32 => 1,
        TensorDType::I8 => 2,
        TensorDType::U8 => 3,
    }
}
fn dtype_from_u8(v: u8) -> Result<TensorDType, WireError> {
    Ok(match v {
        0 => TensorDType::F16,
        1 => TensorDType::F32,
        2 => TensorDType::I8,
        3 => TensorDType::U8,
        _ => return Err(WireError::BadTag),
    })
}

fn layout_to_u8(l: TensorLayout) -> u8 {
    match l {
        TensorLayout::Nchw => 0,
        TensorLayout::Nhwc => 1,
    }
}
fn layout_from_u8(v: u8) -> Result<TensorLayout, WireError> {
    Ok(match v {
        0 => TensorLayout::Nchw,
        1 => TensorLayout::Nhwc,
        _ => return Err(WireError::BadTag),
    })
}

// ---- Dim / Rate ----

fn put_dim(w: &mut Writer, d: &Dim) {
    match d {
        Dim::Any => w.u8(0),
        Dim::Range { min, max } => {
            w.u8(1);
            w.u32(*min);
            w.u32(*max);
        }
        Dim::Fixed(v) => {
            w.u8(2);
            w.u32(*v);
        }
    }
}
fn get_dim(r: &mut Reader) -> Result<Dim, WireError> {
    Ok(match r.u8()? {
        0 => Dim::Any,
        1 => Dim::Range {
            min: r.u32()?,
            max: r.u32()?,
        },
        2 => Dim::Fixed(r.u32()?),
        _ => return Err(WireError::BadTag),
    })
}

fn put_rate(w: &mut Writer, rt: &Rate) {
    match rt {
        Rate::Any => w.u8(0),
        Rate::Range { min_q16, max_q16 } => {
            w.u8(1);
            w.u32(*min_q16);
            w.u32(*max_q16);
        }
        Rate::Fixed(v) => {
            w.u8(2);
            w.u32(*v);
        }
    }
}
fn get_rate(r: &mut Reader) -> Result<Rate, WireError> {
    Ok(match r.u8()? {
        0 => Rate::Any,
        1 => Rate::Range {
            min_q16: r.u32()?,
            max_q16: r.u32()?,
        },
        2 => Rate::Fixed(r.u32()?),
        _ => return Err(WireError::BadTag),
    })
}

// ---- Caps ----

fn put_caps(w: &mut Writer, c: &Caps) {
    match c {
        Caps::CompressedVideo {
            codec,
            width,
            height,
            framerate,
        } => {
            w.u8(0);
            w.u8(video_codec_to_u8(*codec));
            put_dim(w, width);
            put_dim(w, height);
            put_rate(w, framerate);
        }
        Caps::RawVideo {
            format,
            width,
            height,
            framerate,
        } => {
            w.u8(1);
            w.u8(raw_format_to_u8(*format));
            put_dim(w, width);
            put_dim(w, height);
            put_rate(w, framerate);
        }
        Caps::Audio {
            format,
            channels,
            sample_rate,
        } => {
            w.u8(2);
            w.u8(audio_format_to_u8(*format));
            w.u8(*channels);
            w.u32(*sample_rate);
        }
        Caps::Tensor {
            dtype,
            shape,
            layout,
        } => {
            w.u8(3);
            w.u8(dtype_to_u8(*dtype));
            w.u32(shape.dims().len() as u32);
            for d in shape.dims() {
                w.u32(*d);
            }
            w.u8(layout_to_u8(*layout));
        }
        Caps::ByteStream { encoding } => {
            w.u8(4);
            w.u8(bytestream_to_u8(*encoding));
        }
        Caps::Text { format } => {
            w.u8(5);
            w.u8(text_format_to_u8(*format));
        }
    }
}

fn get_caps(r: &mut Reader) -> Result<Caps, WireError> {
    Ok(match r.u8()? {
        0 => Caps::CompressedVideo {
            codec: video_codec_from_u8(r.u8()?)?,
            width: get_dim(r)?,
            height: get_dim(r)?,
            framerate: get_rate(r)?,
        },
        1 => Caps::RawVideo {
            format: raw_format_from_u8(r.u8()?)?,
            width: get_dim(r)?,
            height: get_dim(r)?,
            framerate: get_rate(r)?,
        },
        2 => Caps::Audio {
            format: audio_format_from_u8(r.u8()?)?,
            channels: r.u8()?,
            sample_rate: r.u32()?,
        },
        3 => {
            let dtype = dtype_from_u8(r.u8()?)?;
            // The rank is attacker-controlled; a fixed-rank TensorShape can
            // only carry 1..=MAX_TENSOR_RANK dims, so reject anything else
            // before reading (which also bounds the read loop).
            let n = r.u32()? as usize;
            let mut dims = [0u32; MAX_TENSOR_RANK];
            let slots = dims.get_mut(..n).ok_or(WireError::BadTag)?;
            for d in slots.iter_mut() {
                *d = r.u32()?;
            }
            let layout = layout_from_u8(r.u8()?)?;
            let shape = TensorShape::from_slice(&dims[..n]).ok_or(WireError::BadTag)?;
            Caps::Tensor {
                dtype,
                shape,
                layout,
            }
        }
        4 => Caps::ByteStream {
            encoding: bytestream_from_u8(r.u8()?)?,
        },
        5 => Caps::Text {
            format: text_format_from_u8(r.u8()?)?,
        },
        _ => return Err(WireError::BadTag),
    })
}

// ---- FrameTiming ----

fn put_timing(w: &mut Writer, t: &FrameTiming) {
    w.u64(t.pts_ns);
    w.u64(t.dts_ns);
    w.u64(t.duration_ns);
    w.u64(t.capture_ns);
    w.u64(t.arrival_ns);
    w.bool(t.keyframe);
}
fn get_timing(r: &mut Reader) -> Result<FrameTiming, WireError> {
    Ok(FrameTiming {
        pts_ns: r.u64()?,
        dts_ns: r.u64()?,
        duration_ns: r.u64()?,
        capture_ns: r.u64()?,
        arrival_ns: r.u64()?,
        keyframe: r.bool()?,
    })
}

// ---- Segment ----

fn put_segment(w: &mut Writer, s: &Segment) {
    w.f64(s.rate);
    w.f64(s.applied_rate);
    w.u64(s.base);
    w.u64(s.start);
    match s.stop {
        Some(v) => {
            w.bool(true);
            w.u64(v);
        }
        None => w.bool(false),
    }
    w.u64(s.time);
    w.u64(s.position);
    w.bool(s.key_units_only);
}
fn get_segment(r: &mut Reader) -> Result<Segment, WireError> {
    let rate = r.f64()?;
    let applied_rate = r.f64()?;
    let base = r.u64()?;
    let start = r.u64()?;
    let stop = if r.bool()? { Some(r.u64()?) } else { None };
    let time = r.u64()?;
    let position = r.u64()?;
    let key_units_only = r.bool()?;
    Ok(Segment {
        rate,
        applied_rate,
        base,
        start,
        stop,
        time,
        position,
        key_units_only,
    })
}

// ---- MemoryDomain (CPU only) ----

fn put_domain(w: &mut Writer, d: &MemoryDomain) -> Result<(), WireError> {
    match d {
        MemoryDomain::System(s) => {
            w.u8(DOMAIN_SYSTEM);
            w.bytes(s.as_slice());
            Ok(())
        }
        // A strided shared-CPU view is materialized to dense row-major bytes:
        // the far side receives plain System bytes (the one copy leaving the
        // process costs).
        MemoryDomain::SystemView(v) => {
            w.u8(DOMAIN_SYSTEM);
            let dense = v.materialize();
            w.bytes(&dense);
            Ok(())
        }
        // Everything else is a device / foreign pointer that cannot be shipped.
        _ => Err(WireError::UnsupportedDomain),
    }
}
fn get_domain(r: &mut Reader) -> Result<MemoryDomain, WireError> {
    match r.u8()? {
        DOMAIN_SYSTEM => {
            let bytes = r.bytes()?;
            Ok(MemoryDomain::System(SystemSlice::from_boxed(
                bytes.into_boxed_slice(),
            )))
        }
        _ => Err(WireError::BadTag),
    }
}

// ---- per-frame metadata (last field of a DataFrame body) ----

#[cfg(feature = "metadata")]
fn put_meta(w: &mut Writer, meta: &FrameMetaSet) {
    use crate::meta::{AnalyticsMeta, AnalyticsNode, BlobMeta};

    let analytics = meta.get::<AnalyticsMeta>();
    let blob = meta.get::<BlobMeta>();
    let count = analytics.is_some() as u8 + blob.is_some() as u8;
    w.u8(count);

    if let Some(a) = analytics {
        w.u8(META_ANALYTICS);
        w.u32(a.nodes.len() as u32);
        for node in &a.nodes {
            match node {
                AnalyticsNode::Detection(d) => {
                    w.u8(0);
                    w.f32(d.bbox.x);
                    w.f32(d.bbox.y);
                    w.f32(d.bbox.w);
                    w.f32(d.bbox.h);
                    w.u32(d.label);
                    w.f32(d.confidence);
                }
                AnalyticsNode::Classification(c) => {
                    w.u8(1);
                    w.u32(c.label);
                    w.f32(c.confidence);
                }
                AnalyticsNode::Tracking(t) => {
                    w.u8(2);
                    w.u64(t.object_id);
                }
            }
        }
        w.u32(a.relations.len() as u32);
        for rel in &a.relations {
            w.u32(rel.from as u32);
            w.u32(rel.to as u32);
            w.u8(relation_kind_to_u8(rel.kind));
        }
    }

    if let Some(b) = blob {
        w.u8(META_BLOB);
        w.u32(b.blobs.len() as u32);
        for blob in &b.blobs {
            w.str(&blob.header);
            w.bytes(&blob.payload);
        }
    }
}

#[cfg(not(feature = "metadata"))]
fn put_meta(w: &mut Writer, _meta: &FrameMetaSet) {
    // The baseline `FrameMetaSet` is a ZST: nothing to carry.
    w.u8(0);
}

#[cfg(feature = "metadata")]
fn relation_kind_to_u8(k: crate::meta::RelationKind) -> u8 {
    use crate::meta::RelationKind;
    match k {
        RelationKind::Classifies => 0,
        RelationKind::Tracks => 1,
        RelationKind::Contains => 2,
    }
}

#[cfg(feature = "metadata")]
fn relation_kind_from_u8(v: u8) -> Result<crate::meta::RelationKind, WireError> {
    use crate::meta::RelationKind;
    Ok(match v {
        0 => RelationKind::Classifies,
        1 => RelationKind::Tracks,
        2 => RelationKind::Contains,
        _ => return Err(WireError::BadTag),
    })
}

#[cfg(feature = "metadata")]
fn get_meta(r: &mut Reader) -> Result<FrameMetaSet, WireError> {
    use crate::meta::{
        AnalyticsMeta, AnalyticsNode, BBox, Blob, BlobMeta, Classification, ObjectDetection,
        Relation, Tracking,
    };

    let count = r.u8()?;
    let mut set = FrameMetaSet::new();
    for _ in 0..count {
        match r.u8()? {
            META_ANALYTICS => {
                let mut a = AnalyticsMeta::new();
                let n = r.u32()? as usize;
                for _ in 0..n {
                    let node = match r.u8()? {
                        0 => AnalyticsNode::Detection(ObjectDetection {
                            bbox: BBox {
                                x: r.f32()?,
                                y: r.f32()?,
                                w: r.f32()?,
                                h: r.f32()?,
                            },
                            label: r.u32()?,
                            confidence: r.f32()?,
                        }),
                        1 => AnalyticsNode::Classification(Classification {
                            label: r.u32()?,
                            confidence: r.f32()?,
                        }),
                        2 => AnalyticsNode::Tracking(Tracking {
                            object_id: r.u64()?,
                        }),
                        _ => return Err(WireError::BadTag),
                    };
                    a.nodes.push(node);
                }
                let m = r.u32()? as usize;
                for _ in 0..m {
                    a.relations.push(Relation {
                        from: r.u32()? as usize,
                        to: r.u32()? as usize,
                        kind: relation_kind_from_u8(r.u8()?)?,
                    });
                }
                set.attach(a);
            }
            META_BLOB => {
                let mut b = BlobMeta::new();
                let n = r.u32()? as usize;
                for _ in 0..n {
                    b.blobs.push(Blob {
                        header: r.str()?,
                        payload: r.bytes()?,
                    });
                }
                set.attach(b);
            }
            _ => return Err(WireError::BadTag),
        }
    }
    Ok(set)
}

#[cfg(not(feature = "metadata"))]
fn get_meta(r: &mut Reader) -> Result<FrameMetaSet, WireError> {
    // Metadata is the last field of a DataFrame body, so a `metadata`-off
    // receiver just reads the entry count and ignores the payload that follows
    // (a `metadata`-on peer's metas): the stream is already fully framed by the
    // transport, so the un-consumed tail is harmless. Degrades to no metadata,
    // never to a mis-parse.
    let _count = r.u8()?;
    Ok(FrameMetaSet::new())
}

// ---- public API ----

/// Serialize a [`PipelinePacket`] into a self-contained byte buffer.
///
/// Returns [`WireError::UnsupportedDomain`] for a `DataFrame` whose memory is
/// device-resident or foreign (only [`MemoryDomain::System`] / `SystemView`
/// can cross a byte transport). The transport is expected to length-frame the
/// returned buffer (the codec produces the body only).
pub fn encode_packet(packet: &PipelinePacket) -> Result<Vec<u8>, WireError> {
    let mut w = Writer::new();
    w.u8(WIRE_VERSION);
    match packet {
        PipelinePacket::CapsChanged(caps) => {
            w.u8(PKT_CAPS_CHANGED);
            put_caps(&mut w, caps);
        }
        PipelinePacket::DataFrame(frame) => {
            w.u8(PKT_DATA_FRAME);
            put_timing(&mut w, &frame.timing);
            w.u64(frame.sequence);
            put_domain(&mut w, &frame.domain)?;
            put_meta(&mut w, &frame.meta);
        }
        PipelinePacket::Eos => w.u8(PKT_EOS),
        PipelinePacket::Flush => w.u8(PKT_FLUSH),
        PipelinePacket::Segment(seg) => {
            w.u8(PKT_SEGMENT);
            put_segment(&mut w, seg);
        }
    }
    Ok(w.buf)
}

/// Reconstruct a [`PipelinePacket`] from bytes produced by [`encode_packet`].
///
/// Trailing bytes after the packet are ignored (the transport frames each
/// message), so a `metadata`-on sender's meta payload does not trip a
/// `metadata`-off receiver.
pub fn decode_packet(bytes: &[u8]) -> Result<PipelinePacket, WireError> {
    let mut r = Reader::new(bytes);
    if r.u8()? != WIRE_VERSION {
        return Err(WireError::BadTag);
    }
    Ok(match r.u8()? {
        PKT_CAPS_CHANGED => PipelinePacket::CapsChanged(get_caps(&mut r)?),
        PKT_DATA_FRAME => {
            let timing = get_timing(&mut r)?;
            let sequence = r.u64()?;
            let domain = get_domain(&mut r)?;
            let meta = get_meta(&mut r)?;
            PipelinePacket::DataFrame(Frame {
                domain,
                timing,
                sequence,
                meta,
            })
        }
        PKT_EOS => PipelinePacket::Eos,
        PKT_FLUSH => PipelinePacket::Flush,
        PKT_SEGMENT => PipelinePacket::Segment(get_segment(&mut r)?),
        _ => return Err(WireError::BadTag),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;

    fn roundtrip(p: &PipelinePacket) -> PipelinePacket {
        let bytes = encode_packet(p).expect("encode");
        decode_packet(&bytes).expect("decode")
    }

    #[test]
    fn caps_changed_round_trips_every_variant() {
        let cases = [
            Caps::CompressedVideo {
                codec: VideoCodec::H265,
                width: Dim::Fixed(1920),
                height: Dim::Range {
                    min: 480,
                    max: 1080,
                },
                framerate: Rate::Fixed(30 << 16),
            },
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Any,
            },
            Caps::Audio {
                format: AudioFormat::Opus,
                channels: 2,
                sample_rate: 48_000,
            },
            Caps::Tensor {
                dtype: TensorDType::F32,
                shape: TensorShape::new([1, 3, 224, 224]),
                layout: TensorLayout::Nchw,
            },
            Caps::ByteStream {
                encoding: ByteStreamEncoding::MpegTs,
            },
            Caps::Text {
                format: TextFormat::WebVtt,
            },
        ];
        for caps in cases {
            let p = PipelinePacket::CapsChanged(caps.clone());
            match roundtrip(&p) {
                PipelinePacket::CapsChanged(got) => assert_eq!(got, caps),
                other => panic!("expected CapsChanged, got {other:?}"),
            }
        }
    }

    #[test]
    fn tensor_caps_rank_beyond_max_rejected() {
        // Hand-encode a tensor caps blob whose declared rank exceeds
        // MAX_TENSOR_RANK: the decoder must reject it up front (fixed-rank
        // TensorShape, M636) instead of reading an unbounded dim list.
        let mut w = Writer::new();
        w.u8(3); // Caps::Tensor tag
        w.u8(dtype_to_u8(TensorDType::F32));
        let n = (MAX_TENSOR_RANK + 1) as u32;
        w.u32(n);
        for _ in 0..n {
            w.u32(1);
        }
        w.u8(layout_to_u8(TensorLayout::Nchw));
        let mut r = Reader::new(&w.buf);
        assert_eq!(get_caps(&mut r), Err(WireError::BadTag));
    }

    #[test]
    fn mpeg4_part2_codec_round_trips() {
        let caps = Caps::CompressedVideo {
            codec: VideoCodec::Mpeg4Part2,
            width: Dim::Fixed(720),
            height: Dim::Fixed(576),
            framerate: Rate::Fixed(25 << 16),
        };
        match roundtrip(&PipelinePacket::CapsChanged(caps.clone())) {
            PipelinePacket::CapsChanged(got) => assert_eq!(got, caps),
            other => panic!("expected CapsChanged, got {other:?}"),
        }
        // The wire tag is stable: appended after the existing codecs (Mjpeg = 5).
        assert_eq!(video_codec_to_u8(VideoCodec::Mpeg4Part2), 6);
    }

    #[test]
    fn data_frame_round_trips_bytes_timing_and_sequence() {
        let bytes: Vec<u8> = (0u8..=200).collect();
        let timing = FrameTiming {
            pts_ns: 1_000,
            dts_ns: 900,
            duration_ns: 33,
            capture_ns: 7,
            arrival_ns: 42,
            keyframe: true,
        };
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.clone().into_boxed_slice())),
            timing,
            sequence: 12_345,
            meta: FrameMetaSet::new(),
        };
        match roundtrip(&PipelinePacket::DataFrame(frame)) {
            PipelinePacket::DataFrame(got) => {
                assert_eq!(got.sequence, 12_345);
                assert_eq!(got.timing, timing);
                match got.domain {
                    MemoryDomain::System(s) => assert_eq!(s.as_slice(), &bytes[..]),
                    other => panic!("expected System, got {other:?}"),
                }
            }
            other => panic!("expected DataFrame, got {other:?}"),
        }
    }

    #[test]
    fn control_packets_round_trip() {
        assert!(matches!(
            roundtrip(&PipelinePacket::Eos),
            PipelinePacket::Eos
        ));
        assert!(matches!(
            roundtrip(&PipelinePacket::Flush),
            PipelinePacket::Flush
        ));
        let seg = Segment {
            rate: 2.0,
            applied_rate: 1.0,
            base: 5,
            start: 1_000,
            stop: Some(9_000),
            time: 1_000,
            position: 3_000,
            key_units_only: true,
        };
        match roundtrip(&PipelinePacket::Segment(seg)) {
            PipelinePacket::Segment(got) => assert_eq!(got, seg),
            other => panic!("expected Segment, got {other:?}"),
        }
    }

    #[test]
    fn device_domain_cannot_be_serialized() {
        // A DMABUF is a device fd, not CPU bytes: encoding must refuse it rather
        // than ship a meaningless pointer. (fd -1 never opens a real resource;
        // its Drop close is harmless.)
        // SAFETY: fd -1 is never a live DMABUF; `from_raw` only stores it (no
        // I/O), and the Drop `close(-1)` is a harmless no-op. This exercises the
        // encode refusal of a device domain, not real DMABUF handling.
        let dmabuf = unsafe { crate::memory::OwnedDmaBuf::from_raw(-1, 0, 0) };
        let frame = Frame::new(MemoryDomain::DmaBuf(dmabuf), FrameTiming::default(), 0);
        assert_eq!(
            encode_packet(&PipelinePacket::DataFrame(frame)),
            Err(WireError::UnsupportedDomain)
        );
    }

    #[test]
    fn truncated_and_bad_version_are_rejected() {
        assert!(matches!(decode_packet(&[]), Err(WireError::Truncated)));
        // Wrong version byte.
        assert!(matches!(
            decode_packet(&[WIRE_VERSION + 1, PKT_EOS]),
            Err(WireError::BadTag)
        ));
        // Right version, unknown packet tag.
        assert!(matches!(
            decode_packet(&[WIRE_VERSION, 250]),
            Err(WireError::BadTag)
        ));
        // A CapsChanged header with the caps body cut off.
        let mut bytes = encode_packet(&PipelinePacket::CapsChanged(Caps::Text {
            format: TextFormat::Utf8,
        }))
        .unwrap();
        bytes.pop();
        assert!(matches!(decode_packet(&bytes), Err(WireError::Truncated)));
    }

    #[test]
    fn system_view_frame_materializes_to_system_bytes() {
        use crate::memory::SystemView;
        use crate::tensor::TensorView;
        // A contiguous 1-D view over 8 bytes: materialize is identity here, but
        // it proves a SystemView frame serializes as System bytes.
        let backing: alloc::sync::Arc<[u8]> = Box::<[u8]>::from([1u8, 2, 3, 4, 5, 6, 7, 8]).into();
        let view = TensorView::contiguous(TensorDType::U8, &[8]);
        let frame = Frame::new(
            MemoryDomain::SystemView(SystemView::new(backing, view)),
            FrameTiming::default(),
            1,
        );
        match roundtrip(&PipelinePacket::DataFrame(frame)) {
            PipelinePacket::DataFrame(got) => match got.domain {
                MemoryDomain::System(s) => assert_eq!(s.as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8]),
                other => panic!("SystemView should decode as System, got {other:?}"),
            },
            other => panic!("expected DataFrame, got {other:?}"),
        }
    }

    #[cfg(feature = "metadata")]
    #[test]
    fn analytics_and_blob_metadata_round_trip() {
        use crate::meta::{
            AnalyticsMeta, AnalyticsNode, BBox, BlobMeta, Classification, ObjectDetection,
            RelationKind,
        };
        let mut analytics = AnalyticsMeta::new();
        let d = analytics.add_detection(ObjectDetection {
            bbox: BBox {
                x: 0.1,
                y: 0.2,
                w: 0.3,
                h: 0.4,
            },
            label: 7,
            confidence: 0.9,
        });
        let c = analytics.push(AnalyticsNode::Classification(Classification {
            label: 42,
            confidence: 0.7,
        }));
        analytics.relate(d, c, RelationKind::Classifies);

        let mut blob = BlobMeta::new();
        blob.push("embedding", alloc::vec![1, 2, 3, 4]);

        let mut meta = FrameMetaSet::new();
        meta.attach(analytics.clone());
        meta.attach(blob.clone());

        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([9u8; 16]))),
            timing: FrameTiming::default(),
            sequence: 0,
            meta,
        };
        match roundtrip(&PipelinePacket::DataFrame(frame)) {
            PipelinePacket::DataFrame(got) => {
                let a = got.meta.get::<AnalyticsMeta>().expect("analytics survived");
                assert_eq!(a.nodes, analytics.nodes);
                assert_eq!(a.relations, analytics.relations);
                let b = got.meta.get::<BlobMeta>().expect("blob survived");
                assert_eq!(b, &blob);
            }
            other => panic!("expected DataFrame, got {other:?}"),
        }
    }
}
