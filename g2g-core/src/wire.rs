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
    AudioFormat, ByteStreamEncoding, Caps, ClosedCaptionFormat, Dim, Interlace, Rate,
    RawVideoFormat, SubPictureFormat, TensorDType, TensorLayout, TensorShape, TextFormat,
    VideoCodec,
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
    /// in a string field. Also reported for a packet with no wire tag at all (a
    /// runner-internal [`PipelinePacket::Tick`]).
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
#[cfg_attr(not(feature = "metadata"), allow(dead_code))]
const META_CAPTION: u8 = 2;
#[cfg_attr(not(feature = "metadata"), allow(dead_code))]
const META_HDR_STATIC: u8 = 3;
#[cfg_attr(not(feature = "metadata"), allow(dead_code))]
const META_TIMECODE: u8 = 4;

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
        VideoCodec::SorensonH263 => 8,
        VideoCodec::Vp6 { alpha: false } => 9,
        VideoCodec::Vp6 { alpha: true } => 10,
        VideoCodec::Mpeg2 => 11,
        VideoCodec::Png => 12,
        VideoCodec::WebP => 13,
        VideoCodec::Vc1 => 14,
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
        8 => VideoCodec::SorensonH263,
        9 => VideoCodec::Vp6 { alpha: false },
        10 => VideoCodec::Vp6 { alpha: true },
        11 => VideoCodec::Mpeg2,
        12 => VideoCodec::Png,
        13 => VideoCodec::WebP,
        14 => VideoCodec::Vc1,
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
        RawVideoFormat::P010 => 13,
        RawVideoFormat::Rgb8 => 14,
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
        13 => RawVideoFormat::P010,
        14 => RawVideoFormat::Rgb8,
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
        AudioFormat::Vorbis => 11,
        AudioFormat::Mp3 => 12,
        AudioFormat::Speex => 13,
        AudioFormat::PcmS32Le => 14,
        AudioFormat::PcmU8 => 15,
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
        11 => AudioFormat::Vorbis,
        12 => AudioFormat::Mp3,
        13 => AudioFormat::Speex,
        14 => AudioFormat::PcmS32Le,
        15 => AudioFormat::PcmU8,
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
        ByteStreamEncoding::MpegPs => 7,
        ByteStreamEncoding::Wav => 8,
        ByteStreamEncoding::Aiff => 18,
        ByteStreamEncoding::Au => 19,
        ByteStreamEncoding::Avi => 9,
        ByteStreamEncoding::Y4m => 10,
        ByteStreamEncoding::Multipart => 11,
        ByteStreamEncoding::Raw => 12,
        ByteStreamEncoding::Rtp => 13,
        ByteStreamEncoding::Srtp => 14,
        ByteStreamEncoding::Rtcp => 15,
        ByteStreamEncoding::Srtcp => 16,
        ByteStreamEncoding::Dtls => 17,
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
        7 => ByteStreamEncoding::MpegPs,
        8 => ByteStreamEncoding::Wav,
        9 => ByteStreamEncoding::Avi,
        18 => ByteStreamEncoding::Aiff,
        19 => ByteStreamEncoding::Au,
        10 => ByteStreamEncoding::Y4m,
        11 => ByteStreamEncoding::Multipart,
        12 => ByteStreamEncoding::Raw,
        13 => ByteStreamEncoding::Rtp,
        14 => ByteStreamEncoding::Srtp,
        15 => ByteStreamEncoding::Rtcp,
        16 => ByteStreamEncoding::Srtcp,
        17 => ByteStreamEncoding::Dtls,
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
        TextFormat::Teletext => 6,
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
        6 => TextFormat::Teletext,
        _ => return Err(WireError::BadTag),
    })
}

fn cc_format_to_u8(f: ClosedCaptionFormat) -> u8 {
    match f {
        ClosedCaptionFormat::Cea608 => 0,
        ClosedCaptionFormat::Cea708 => 1,
        ClosedCaptionFormat::Cea608Raw => 2,
        ClosedCaptionFormat::Cea608S334 => 3,
        ClosedCaptionFormat::Cea708Cdp => 4,
    }
}
fn cc_format_from_u8(v: u8) -> Result<ClosedCaptionFormat, WireError> {
    Ok(match v {
        0 => ClosedCaptionFormat::Cea608,
        1 => ClosedCaptionFormat::Cea708,
        2 => ClosedCaptionFormat::Cea608Raw,
        3 => ClosedCaptionFormat::Cea608S334,
        4 => ClosedCaptionFormat::Cea708Cdp,
        _ => return Err(WireError::BadTag),
    })
}

fn subpicture_format_to_u8(f: SubPictureFormat) -> u8 {
    match f {
        SubPictureFormat::VobSub => 0,
        SubPictureFormat::DvbSub => 1,
        SubPictureFormat::Pgs => 2,
    }
}
fn subpicture_format_from_u8(v: u8) -> Result<SubPictureFormat, WireError> {
    Ok(match v {
        0 => SubPictureFormat::VobSub,
        1 => SubPictureFormat::DvbSub,
        2 => SubPictureFormat::Pgs,
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

fn interlace_to_u8(i: Interlace) -> u8 {
    match i {
        Interlace::Any => 0,
        Interlace::Progressive => 1,
        Interlace::Interleaved => 2,
    }
}
fn interlace_from_u8(v: u8) -> Result<Interlace, WireError> {
    Ok(match v {
        0 => Interlace::Any,
        1 => Interlace::Progressive,
        2 => Interlace::Interleaved,
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
            interlace,
        } => {
            w.u8(1);
            w.u8(raw_format_to_u8(*format));
            put_dim(w, width);
            put_dim(w, height);
            put_rate(w, framerate);
            w.u8(interlace_to_u8(*interlace));
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
        Caps::Klv => w.u8(6),
        Caps::ClosedCaption { format } => {
            w.u8(7);
            w.u8(cc_format_to_u8(*format));
        }
        Caps::SubPicture { format } => {
            w.u8(8);
            w.u8(subpicture_format_to_u8(*format));
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
            interlace: interlace_from_u8(r.u8()?)?,
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
        6 => Caps::Klv,
        7 => Caps::ClosedCaption {
            format: cc_format_from_u8(r.u8()?)?,
        },
        8 => Caps::SubPicture {
            format: subpicture_format_from_u8(r.u8()?)?,
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
    use crate::meta::{
        AnalyticsMeta, AnalyticsNode, BlobMeta, CaptionMeta, HdrStaticMeta, TimecodeMeta,
    };

    let analytics = meta.get::<AnalyticsMeta>();
    let blob = meta.get::<BlobMeta>();
    let caption = meta.get::<CaptionMeta>();
    let hdr = meta.get::<HdrStaticMeta>();
    let timecode = meta.get::<TimecodeMeta>();
    let count = analytics.is_some() as u8
        + blob.is_some() as u8
        + caption.is_some() as u8
        + hdr.is_some() as u8
        + timecode.is_some() as u8;
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
                AnalyticsNode::Segmentation(s) => {
                    w.u8(3);
                    w.f32(s.bbox.x);
                    w.f32(s.bbox.y);
                    w.f32(s.bbox.w);
                    w.f32(s.bbox.h);
                    w.u32(s.label);
                    w.f32(s.confidence);
                    w.u32(s.mask.width());
                    w.u32(s.mask.height());
                    w.u32(s.mask.stride());
                    w.bytes(s.mask.data());
                }
                AnalyticsNode::Roi(r) => {
                    w.u8(4);
                    w.f32(r.bbox.x);
                    w.f32(r.bbox.y);
                    w.f32(r.bbox.w);
                    w.f32(r.bbox.h);
                    w.u32(r.id);
                    w.u32(r.label);
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

    if let Some(c) = caption {
        w.u8(META_CAPTION);
        w.u32(c.triples.len() as u32);
        for t in &c.triples {
            w.u8(t.cc_type);
            w.u8(t.b0);
            w.u8(t.b1);
        }
    }

    if let Some(h) = hdr {
        w.u8(META_HDR_STATIC);
        match &h.mastering {
            Some(m) => {
                w.bool(true);
                for p in &m.display_primaries {
                    w.f32(p.x);
                    w.f32(p.y);
                }
                w.f32(m.white_point.x);
                w.f32(m.white_point.y);
                w.f32(m.max_luminance);
                w.f32(m.min_luminance);
            }
            None => w.bool(false),
        }
        put_opt_u16(w, h.max_content_light_level);
        put_opt_u16(w, h.max_frame_average_light_level);
    }

    if let Some(t) = timecode {
        w.u8(META_TIMECODE);
        w.u8(t.hours);
        w.u8(t.minutes);
        w.u8(t.seconds);
        w.u8(t.frames);
        w.bool(t.drop_frame);
        w.bool(t.framerate_q16.is_some());
        w.u32(t.framerate_q16.unwrap_or(0));
    }
}

/// An optional `u16` as a presence flag then the value (only the HDR meta needs
/// one, so it is not a `Writer` primitive).
#[cfg(feature = "metadata")]
fn put_opt_u16(w: &mut Writer, v: Option<u16>) {
    w.bool(v.is_some());
    w.u32(v.unwrap_or(0) as u32);
}

#[cfg(feature = "metadata")]
fn get_opt_u16(r: &mut Reader) -> Result<Option<u16>, WireError> {
    let present = r.bool()?;
    let v = u16::try_from(r.u32()?).map_err(|_| WireError::BadTag)?;
    Ok(present.then_some(v))
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
        AnalyticsMeta, AnalyticsNode, BBox, Blob, BlobMeta, CaptionMeta, CaptionTriple,
        Chromaticity, Classification, HdrStaticMeta, Mask, MasteringDisplay, ObjectDetection,
        Relation, Roi, Segmentation, TimecodeMeta, Tracking,
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
                        3 => {
                            let bbox = BBox {
                                x: r.f32()?,
                                y: r.f32()?,
                                w: r.f32()?,
                                h: r.f32()?,
                            };
                            let label = r.u32()?;
                            let confidence = r.f32()?;
                            let (width, height, stride) = (r.u32()?, r.u32()?, r.u32()?);
                            // The mask bytes are length-prefixed and bounded by
                            // the message, and `Mask::new` rejects geometry that
                            // does not fit them: a peer cannot make us index out
                            // of the buffer it sent.
                            let mask = Mask::new(width, height, stride, r.bytes()?)
                                .ok_or(WireError::BadTag)?;
                            AnalyticsNode::Segmentation(Segmentation {
                                bbox,
                                label,
                                confidence,
                                mask,
                            })
                        }
                        4 => AnalyticsNode::Roi(Roi {
                            bbox: BBox {
                                x: r.f32()?,
                                y: r.f32()?,
                                w: r.f32()?,
                                h: r.f32()?,
                            },
                            id: r.u32()?,
                            label: r.u32()?,
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
            META_CAPTION => {
                let mut c = CaptionMeta::new();
                let n = r.u32()? as usize;
                for _ in 0..n {
                    c.push(CaptionTriple {
                        cc_type: r.u8()?,
                        b0: r.u8()?,
                        b1: r.u8()?,
                    });
                }
                set.attach(c);
            }
            META_HDR_STATIC => {
                let mastering = if r.bool()? {
                    let mut primaries = [Chromaticity { x: 0.0, y: 0.0 }; 3];
                    for p in &mut primaries {
                        p.x = r.f32()?;
                        p.y = r.f32()?;
                    }
                    Some(MasteringDisplay {
                        display_primaries: primaries,
                        white_point: Chromaticity {
                            x: r.f32()?,
                            y: r.f32()?,
                        },
                        max_luminance: r.f32()?,
                        min_luminance: r.f32()?,
                    })
                } else {
                    None
                };
                set.attach(HdrStaticMeta {
                    mastering,
                    max_content_light_level: get_opt_u16(r)?,
                    max_frame_average_light_level: get_opt_u16(r)?,
                });
            }
            META_TIMECODE => {
                let tc = TimecodeMeta {
                    hours: r.u8()?,
                    minutes: r.u8()?,
                    seconds: r.u8()?,
                    frames: r.u8()?,
                    drop_frame: r.bool()?,
                    framerate_q16: {
                        let present = r.bool()?;
                        let v = r.u32()?;
                        present.then_some(v)
                    },
                };
                set.attach(tc);
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
        // A `Tick` is runner-internal (a fan-in arm's deadline, consumed at the
        // arm), so it has no wire tag and cannot reach a transport. There is no
        // skip convention here (every packet encodes to a body), so encoding one
        // is a bug, reported rather than silently dropped.
        PipelinePacket::Tick => return Err(WireError::BadTag),
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

// ---- framed recordings ----

/// Width of a framed record's length prefix: a `u32-le` payload byte count
/// ahead of each [`encode_packet`] body.
pub const RECORD_LENGTH_PREFIX_BYTES: usize = 4;

/// The length prefix that frames a `payload_len`-byte [`encode_packet`] body in
/// a recording, the one definition of the on-disk record framing that
/// `recordsink`, `replaysrc`, and the runner's flight-recorder dump share.
/// [`UnsupportedDomain`](WireError::UnsupportedDomain) for a payload too large
/// to describe in the prefix.
pub fn record_length_prefix(
    payload_len: usize,
) -> Result<[u8; RECORD_LENGTH_PREFIX_BYTES], WireError> {
    let len = u32::try_from(payload_len).map_err(|_| WireError::UnsupportedDomain)?;
    Ok(len.to_le_bytes())
}

/// Split a recording buffer into its packets. A truncated trailing record (a
/// recording cut off mid-write, e.g. by the crash being investigated) is dropped
/// rather than failing the replay.
pub fn read_records(buf: &[u8]) -> Result<Vec<PipelinePacket>, WireError> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + RECORD_LENGTH_PREFIX_BYTES <= buf.len() {
        let len = u32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]) as usize;
        let start = i + RECORD_LENGTH_PREFIX_BYTES;
        let end = match start.checked_add(len) {
            Some(e) if e <= buf.len() => e,
            _ => break, // truncated tail
        };
        out.push(decode_packet(&buf[start..end])?);
        i = end;
    }
    Ok(out)
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
    fn every_codec_tag_round_trips() {
        // A wrong tag would silently retarget a remote stream's codec. A shared
        // tag fails here too: only one variant can come back out of the byte.
        let video = [
            VideoCodec::H264,
            VideoCodec::H265,
            VideoCodec::Av1,
            VideoCodec::Vp8,
            VideoCodec::Vp9,
            VideoCodec::Mjpeg,
            VideoCodec::Mpeg4Part2,
            VideoCodec::JpegXs,
            VideoCodec::SorensonH263,
            VideoCodec::Vp6 { alpha: false },
            VideoCodec::Vp6 { alpha: true },
            VideoCodec::Mpeg2,
            VideoCodec::Png,
            VideoCodec::WebP,
            VideoCodec::Vc1,
        ];
        for c in video {
            assert_eq!(video_codec_from_u8(video_codec_to_u8(c)), Ok(c));
        }
        let audio = [
            AudioFormat::Aac,
            AudioFormat::Opus,
            AudioFormat::Mp2,
            AudioFormat::Mp3,
            AudioFormat::Speex,
            AudioFormat::Ac3,
            AudioFormat::Flac,
            AudioFormat::Vorbis,
            AudioFormat::PcmS16Le,
            AudioFormat::PcmF32Le,
            AudioFormat::PcmS24Le,
            AudioFormat::PcmS32Le,
            AudioFormat::PcmU8,
            AudioFormat::Mulaw,
            AudioFormat::Alaw,
            AudioFormat::ImaAdpcm,
        ];
        for f in audio {
            assert_eq!(audio_format_from_u8(audio_format_to_u8(f)), Ok(f));
        }
    }

    #[test]
    fn packet_bytestream_tags_round_trip() {
        for encoding in [
            ByteStreamEncoding::Rtp,
            ByteStreamEncoding::Srtp,
            ByteStreamEncoding::Rtcp,
            ByteStreamEncoding::Srtcp,
            ByteStreamEncoding::Dtls,
            ByteStreamEncoding::Aiff,
            ByteStreamEncoding::Au,
        ] {
            assert_eq!(bytestream_from_u8(bytestream_to_u8(encoding)), Ok(encoding));
        }
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
                interlace: crate::Interlace::Any,
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
            Caps::ClosedCaption {
                format: ClosedCaptionFormat::Cea708,
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

    #[cfg(feature = "metadata")]
    #[test]
    fn segmentation_and_roi_nodes_round_trip() {
        use crate::meta::{AnalyticsMeta, AnalyticsNode, BBox, Mask, Roi, Segmentation};
        let bbox = BBox {
            x: 0.25,
            y: 0.5,
            w: 0.1,
            h: 0.2,
        };
        // A 3x2 mask with a 4-byte stride, so the padded layout has to survive.
        let mask = Mask::new(3, 2, 4, alloc::vec![10, 20, 30, 0, 40, 50, 60, 0])
            .expect("mask fits its data");
        let mut analytics = AnalyticsMeta::new();
        analytics.push(AnalyticsNode::Segmentation(Segmentation {
            bbox,
            label: 3,
            confidence: 0.75,
            mask,
        }));
        analytics.push(AnalyticsNode::Roi(Roi {
            bbox,
            id: 9,
            label: 4,
        }));

        let mut meta = FrameMetaSet::new();
        meta.attach(analytics.clone());
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
            timing: FrameTiming::default(),
            sequence: 0,
            meta,
        };
        match roundtrip(&PipelinePacket::DataFrame(frame)) {
            PipelinePacket::DataFrame(got) => {
                let a = got.meta.get::<AnalyticsMeta>().expect("analytics survived");
                assert_eq!(a.nodes, analytics.nodes);
                let seg = a.segmentations().next().expect("segmentation node");
                assert_eq!(seg.mask.sample(2, 1), Some(60));
                assert_eq!(seg.mask.sample(3, 0), None, "outside the mask width");
                assert_eq!(a.rois().next().expect("roi node").id, 9);
            }
            other => panic!("expected DataFrame, got {other:?}"),
        }
    }

    #[cfg(feature = "metadata")]
    #[test]
    fn a_mask_whose_geometry_overruns_its_bytes_is_rejected() {
        use crate::meta::Mask;
        assert!(
            Mask::new(4, 4, 4, alloc::vec![0; 15]).is_none(),
            "short data"
        );
        assert!(
            Mask::new(8, 2, 4, alloc::vec![0; 64]).is_none(),
            "stride < width"
        );
        assert!(
            Mask::new(u32::MAX, u32::MAX, u32::MAX, alloc::vec![0; 8]).is_none(),
            "the row product must not overflow into a valid-looking size"
        );
    }

    #[cfg(feature = "metadata")]
    #[test]
    fn caption_metadata_round_trips() {
        use crate::meta::{CaptionMeta, CaptionTriple};
        let mut captions = CaptionMeta::new();
        captions.push(CaptionTriple {
            cc_type: 0,
            b0: 0x94,
            b1: 0xAE,
        });
        captions.push(CaptionTriple {
            cc_type: 3,
            b0: 0x01,
            b1: 0xFF,
        });

        let mut meta = FrameMetaSet::new();
        meta.attach(captions.clone());
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
            timing: FrameTiming::default(),
            sequence: 3,
            meta,
        };
        match roundtrip(&PipelinePacket::DataFrame(frame)) {
            PipelinePacket::DataFrame(got) => {
                let c = got.meta.get::<CaptionMeta>().expect("captions survived");
                assert_eq!(c, &captions);
            }
            other => panic!("expected DataFrame, got {other:?}"),
        }
    }

    #[cfg(feature = "metadata")]
    #[test]
    fn hdr_static_metadata_round_trips() {
        use crate::meta::{Chromaticity, HdrStaticMeta, MasteringDisplay};
        let xy = |x, y| Chromaticity { x, y };
        let hdr = HdrStaticMeta {
            mastering: Some(MasteringDisplay {
                display_primaries: [xy(0.708, 0.292), xy(0.170, 0.797), xy(0.131, 0.046)],
                white_point: xy(0.3127, 0.3290),
                max_luminance: 1000.0,
                min_luminance: 0.005,
            }),
            max_content_light_level: Some(1200),
            max_frame_average_light_level: Some(300),
        };

        let mut meta = FrameMetaSet::new();
        meta.attach(hdr);
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
            timing: FrameTiming::default(),
            sequence: 0,
            meta,
        };
        match roundtrip(&PipelinePacket::DataFrame(frame)) {
            PipelinePacket::DataFrame(got) => {
                let h = got.meta.get::<HdrStaticMeta>().expect("hdr survived");
                assert_eq!(h, &hdr);
            }
            other => panic!("expected DataFrame, got {other:?}"),
        }
    }

    #[cfg(feature = "metadata")]
    #[test]
    fn hdr_static_metadata_round_trips_without_a_mastering_display() {
        // A stream carrying only content_light_level_info: the absent half must
        // decode back as absent, not as zeroed primaries.
        use crate::meta::HdrStaticMeta;
        let hdr = HdrStaticMeta {
            mastering: None,
            max_content_light_level: Some(400),
            max_frame_average_light_level: None,
        };
        let mut meta = FrameMetaSet::new();
        meta.attach(hdr);
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
            timing: FrameTiming::default(),
            sequence: 0,
            meta,
        };
        match roundtrip(&PipelinePacket::DataFrame(frame)) {
            PipelinePacket::DataFrame(got) => {
                assert_eq!(got.meta.get::<HdrStaticMeta>(), Some(&hdr));
            }
            other => panic!("expected DataFrame, got {other:?}"),
        }
    }

    #[cfg(feature = "metadata")]
    #[test]
    fn timecode_metadata_round_trips() {
        use crate::meta::TimecodeMeta;
        let tc = TimecodeMeta {
            hours: 10,
            minutes: 59,
            seconds: 58,
            frames: 29,
            drop_frame: true,
            framerate_q16: Some(1_965_691), // 29.97 fps
        };
        let mut meta = FrameMetaSet::new();
        meta.attach(tc);
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
            timing: FrameTiming::default(),
            sequence: 0,
            meta,
        };
        match roundtrip(&PipelinePacket::DataFrame(frame)) {
            PipelinePacket::DataFrame(got) => {
                assert_eq!(got.meta.get::<TimecodeMeta>(), Some(&tc));
            }
            other => panic!("expected DataFrame, got {other:?}"),
        }
    }
}
