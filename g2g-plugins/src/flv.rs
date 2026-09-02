//! Pure FLV (Flash Video) container parser (M119), the byte-stream sibling of
//! [`crate::mpegts::TsDemuxer`] / [`crate::ogg::OggDemuxer`]. `no_std`, no I/O.
//!
//! FLV is a flat tag stream: a 9-byte header, then `PreviousTagSize` (UI32) /
//! tag pairs. Each tag is an 11-byte header (type, 24-bit data size, 24+8-bit
//! millisecond timestamp, stream id) followed by its body. The body's first byte
//! identifies the codec, and each codec puts a different number of bytes between
//! it and the access unit; [`FlvCodec`] names the ones this parser carries and
//! `payload_offset` holds the widths.
//!
//! Video: H.264 (id 7, AVCC length-prefixed NALUs), Sorenson Spark (id 2),
//! VP6 (id 4) and VP6 with alpha (id 5). Audio: AAC (sound format 10, raw
//! frames), MP3 (format 2, and format 14 for MP3 at 8 kHz) and Speex (format 11,
//! always 16 kHz mono).
//!
//! The sequence-header tags (the `AVCDecoderConfigurationRecord` /
//! `AudioSpecificConfig`) are retained as the codec-config side channel
//! ([`FlvDemuxer::video_config`] / [`FlvDemuxer::audio_config`], M662) rather
//! than emitted as units, so the element can convert the AVCC media frames to a
//! self-describing elementary stream; VP6's one-byte dimension adjustment rides
//! the same channel (it is what libavcodec wants as extradata). The `onMetaData`
//! script tag's body is retained ([`FlvDemuxer::metadata`]) so the element can
//! surface its AMF0 metadata via the tag system.

use alloc::vec::Vec;

use g2g_core::TagList;

/// FLV tag type: an audio tag (codec-tagged audio data).
const TAG_AUDIO: u8 = 8;
/// FLV tag type: a video tag (codec-tagged video data).
const TAG_VIDEO: u8 = 9;
/// FLV tag type: a script-data tag (AMF, carries `onMetaData`).
const TAG_SCRIPT: u8 = 18;

/// FLV video codec id for AVC / H.264 (the low nibble of a video tag's first
/// byte).
const VIDEO_CODEC_AVC: u8 = 7;
/// FLV audio sound format for AAC (the high nibble of an audio tag's first byte).
const SOUND_FORMAT_AAC: u8 = 10;

/// A codec an FLV tag can carry, in the form the tag's first byte identifies it:
/// a video codec id (low nibble of a video tag) or an audio sound format (high
/// nibble of an audio tag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlvCodec {
    /// Sorenson Spark (video codec id 2).
    SorensonH263,
    /// On2 VP6, Flash variant (video codec id 4).
    Vp6,
    /// On2 VP6 with an alpha plane (video codec id 5).
    Vp6Alpha,
    /// H.264 / AVC (video codec id 7).
    H264,
    /// MP3 (audio sound format 2, or 14 for the 8 kHz variant).
    Mp3,
    /// AAC (audio sound format 10).
    Aac,
    /// Speex (audio sound format 11), fixed at 16 kHz mono.
    Speex,
}

impl FlvCodec {
    /// Which of FLV's two elementary streams this codec belongs to.
    pub fn track(self) -> FlvTrack {
        match self {
            FlvCodec::SorensonH263 | FlvCodec::Vp6 | FlvCodec::Vp6Alpha | FlvCodec::H264 => {
                FlvTrack::Video
            }
            FlvCodec::Mp3 | FlvCodec::Aac | FlvCodec::Speex => FlvTrack::Audio,
        }
    }

    /// The video codec id / audio sound format written into a tag's first byte.
    /// MP3 always writes format 2: format 14 (MP3 at 8 kHz) is readable but not
    /// written, since libavcodec's FLV demuxer does not recognize it.
    fn tag_id(self) -> u8 {
        match self {
            FlvCodec::SorensonH263 => 2,
            FlvCodec::Vp6 => 4,
            FlvCodec::Vp6Alpha => 5,
            FlvCodec::H264 => VIDEO_CODEC_AVC,
            FlvCodec::Mp3 => 2,
            FlvCodec::Aac => SOUND_FORMAT_AAC,
            FlvCodec::Speex => 11,
        }
    }

    /// The video codec for a video tag's codec id, or `None` for one this parser
    /// does not carry (screen video, MPEG-4 part 2, the enhanced-FLV fourccs).
    fn from_video_id(id: u8) -> Option<Self> {
        match id {
            2 => Some(FlvCodec::SorensonH263),
            4 => Some(FlvCodec::Vp6),
            5 => Some(FlvCodec::Vp6Alpha),
            VIDEO_CODEC_AVC => Some(FlvCodec::H264),
            _ => None,
        }
    }

    /// The audio codec for an audio tag's sound format, or `None` for one this
    /// parser does not carry (PCM, ADPCM, Nellymoser, G.711).
    fn from_sound_format(format: u8) -> Option<Self> {
        match format {
            // 14 is MP3 at 8 kHz: the same bitstream, a rate the 2-bit rate field
            // cannot express.
            2 | 14 => Some(FlvCodec::Mp3),
            SOUND_FORMAT_AAC => Some(FlvCodec::Aac),
            11 => Some(FlvCodec::Speex),
            _ => None,
        }
    }
}

/// Bytes between a tag's first byte and the access unit, per codec. H.264 spends
/// 4 (packet type + 24-bit composition offset); VP6 and VP6-alpha spend 1 (the
/// dimension adjustment nibbles, which libavcodec takes as extradata; VP6-alpha's
/// 24-bit offset to the alpha plane stays in the payload, where its decoder reads
/// it); AAC spends 1 (the packet type); the rest none.
fn payload_offset(codec: FlvCodec) -> usize {
    match codec {
        FlvCodec::H264 => 4,
        FlvCodec::Vp6 | FlvCodec::Vp6Alpha => 1,
        FlvCodec::Aac => 1,
        FlvCodec::SorensonH263 | FlvCodec::Mp3 | FlvCodec::Speex => 0,
    }
}

/// Channel count and sample rate an audio tag's first byte declares, for the
/// codecs whose rate is not carried in a decoder config. The 2-bit rate field
/// counts 5512 / 11025 / 22050 / 44100 Hz; sound format 14 overrides it with
/// 8 kHz and Speex is always 16 kHz mono, matching libavcodec's FLV demuxer.
/// `None` for AAC, whose real layout comes from the `AudioSpecificConfig`.
fn audio_tag_params(codec: FlvCodec, first: u8) -> Option<(u8, u32)> {
    let channels = if first & 0x01 == 1 { 2 } else { 1 };
    let rate = 44_100u32 >> (3 - ((first >> 2) & 0x03));
    match codec {
        FlvCodec::Mp3 if first >> 4 == 14 => Some((channels, 8_000)),
        FlvCodec::Mp3 => Some((channels, rate)),
        FlvCodec::Speex => Some((1, 16_000)),
        FlvCodec::Aac => None,
        _ => None,
    }
}

/// The first byte of an audio tag: sound format, then the rate / sample-size /
/// channel flags. AAC pins the flags at 44 kHz 16-bit stereo (the FLV spec's
/// requirement, the real layout being in the `AudioSpecificConfig`) and Speex at
/// 11 kHz 16-bit mono (what libavcodec writes, its real rate being fixed at
/// 16 kHz); MP3 declares the nearest expressible rate at or below its own.
fn audio_tag_flags(codec: FlvCodec, channels: u8, sample_rate: u32) -> u8 {
    let (rate_bits, stereo) = match codec {
        FlvCodec::Aac => (3, 1),
        FlvCodec::Speex => (1, 0),
        _ => {
            // 44100 >> (3 - bits): the largest declared rate not above the real one.
            let bits = (0..=3u8)
                .rev()
                .find(|&b| 44_100u32 >> (3 - b) <= sample_rate)
                .unwrap_or(0);
            (bits, u8::from(channels >= 2))
        }
    };
    (codec.tag_id() << 4) | (rate_bits << 2) | 0x02 | stereo
}

/// The FLV header (`FLV` signature + version + flags) plus the first
/// `PreviousTagSize0`; `data_offset` (header bytes) is read from the header.
const FLV_HEADER_MIN: usize = 9;
/// Bytes of an FLV tag header before the body: type(1) + data size(3) +
/// timestamp(3) + timestamp extension(1) + stream id(3).
const TAG_HEADER_LEN: usize = 11;
/// The `PreviousTagSize` (UI32) that prefixes every tag after the header.
const PREV_TAG_SIZE_LEN: usize = 4;

// AMF0 markers the `onMetaData` writer emits (the inverse of the reader subset in
// `flvdemux::amf0`).
const AMF0_STRING: u8 = 0x02;
const AMF0_ECMA_ARRAY: u8 = 0x08;
const AMF0_OBJECT_END: u8 = 0x09;

/// Which elementary stream an [`FlvUnit`] belongs to. An FLV stream interleaves
/// at most one video and one audio track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlvTrack {
    Video,
    Audio,
}

/// One demuxed access unit: the codec it came from, its payload (AVCC NALUs for
/// H.264, a raw frame for every other codec), and its millisecond timestamps.
/// The FLV tag timestamp is the decode time; a video tag's signed
/// composition-time offset yields the presentation time (`pts = dts + cts`,
/// M662), so B-frame streams carry both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlvUnit {
    pub codec: FlvCodec,
    pub data: Vec<u8>,
    pub pts_ms: u32,
    pub dts_ms: u32,
    /// Whether this is a resync point: a video keyframe (FLV frame type 1) or any
    /// audio frame. Used by the demuxer seek path (M362) to snap to a decodable
    /// resume point.
    pub keyframe: bool,
}

impl FlvUnit {
    /// Which elementary stream this unit belongs to.
    pub fn track(&self) -> FlvTrack {
        self.codec.track()
    }
}

/// Incremental FLV demuxer: feed bytes with [`push_data`](Self::push_data), drain
/// completed access units with [`take_units`](Self::take_units).
#[derive(Debug, Default)]
pub struct FlvDemuxer {
    buf: Vec<u8>,
    header_done: bool,
    units: Vec<FlvUnit>,
    /// The first `onMetaData` script-tag body, kept so the element can parse its
    /// AMF0 metadata into tags. `None` until a script tag is seen.
    metadata: Option<Vec<u8>>,
    /// The video sequence-header body (the `AVCDecoderConfigurationRecord`),
    /// updated on every sequence-header tag so a mid-stream config change wins.
    video_config: Option<Vec<u8>>,
    /// The audio sequence-header body (the AAC `AudioSpecificConfig`).
    audio_config: Option<Vec<u8>>,
    /// Channels / sample rate the first audio tag's flags declared, for the
    /// codecs that carry no decoder config (M831).
    audio_params: Option<(u8, u32)>,
}

impl FlvDemuxer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append input bytes and parse as many whole tags as are now available.
    pub fn push_data(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
        self.parse();
    }

    /// Take the access units parsed so far, leaving the demuxer ready for more.
    pub fn take_units(&mut self) -> Vec<FlvUnit> {
        core::mem::take(&mut self.units)
    }

    /// The `onMetaData` script-tag body (AMF0), once a script tag has been seen.
    pub fn metadata(&self) -> Option<&[u8]> {
        self.metadata.as_deref()
    }

    /// The video decoder config (the `AVCDecoderConfigurationRecord` from the
    /// AVC sequence-header tag), once seen (M662).
    pub fn video_config(&self) -> Option<&[u8]> {
        self.video_config.as_deref()
    }

    /// The audio decoder config (the AAC `AudioSpecificConfig` from the AAC
    /// sequence-header tag), once seen (M662).
    pub fn audio_config(&self) -> Option<&[u8]> {
        self.audio_config.as_deref()
    }

    /// Channels and sample rate the audio tags declare, for the codecs whose
    /// layout is not in a decoder config (MP3, Speex). `None` before the first
    /// audio tag, and for AAC (whose layout is in the `AudioSpecificConfig`).
    pub fn audio_params(&self) -> Option<(u8, u32)> {
        self.audio_params
    }

    /// Consume the header (once) and every complete `PreviousTagSize` + tag
    /// record from the buffer, appending the access units of supported codecs.
    fn parse(&mut self) {
        let mut pos = 0;
        if !self.header_done {
            if self.buf.len() < FLV_HEADER_MIN {
                return;
            }
            if &self.buf[0..3] != b"FLV" {
                // Not an FLV stream; the caps said otherwise, so drop the bytes
                // rather than spin forever on a header that will never match.
                self.buf.clear();
                return;
            }
            // The header's declared length (>= 9); the body follows it.
            let data_offset =
                u32::from_be_bytes([self.buf[5], self.buf[6], self.buf[7], self.buf[8]]) as usize;
            let data_offset = data_offset.max(FLV_HEADER_MIN);
            if self.buf.len() < data_offset {
                return;
            }
            pos = data_offset;
            self.header_done = true;
        }

        // Each record is a `PreviousTagSize` prefix (PreviousTagSize0 prefixes the
        // first tag, PreviousTagSize_i prefixes tag i+1) then an 11-byte tag
        // header and its body, so the final tag needs no trailing bytes.
        let mut units = Vec::new();
        let mut metadata: Option<Vec<u8>> = None;
        let mut video_config: Option<Vec<u8>> = None;
        let mut audio_config: Option<Vec<u8>> = None;
        let mut audio_params: Option<(u8, u32)> = None;
        loop {
            let header = pos + PREV_TAG_SIZE_LEN;
            if header + TAG_HEADER_LEN > self.buf.len() {
                break;
            }
            let tag_type = self.buf[header] & 0x1F;
            let data_size = ((self.buf[header + 1] as usize) << 16)
                | ((self.buf[header + 2] as usize) << 8)
                | self.buf[header + 3] as usize;
            let ts_lower = ((self.buf[header + 4] as u32) << 16)
                | ((self.buf[header + 5] as u32) << 8)
                | self.buf[header + 6] as u32;
            let timestamp = ((self.buf[header + 7] as u32) << 24) | ts_lower;

            let body_start = header + TAG_HEADER_LEN;
            let body_end = body_start + data_size;
            if body_end > self.buf.len() {
                break; // tag body not fully arrived yet
            }
            let body = &self.buf[body_start..body_end];
            if tag_type == TAG_SCRIPT && metadata.is_none() {
                metadata = Some(body.to_vec());
            } else if let Some((track, config)) = sequence_header(tag_type, body) {
                match track {
                    FlvTrack::Video => video_config = Some(config.to_vec()),
                    FlvTrack::Audio => audio_config = Some(config.to_vec()),
                }
            } else if let Some(unit) = parse_tag(tag_type, timestamp, body) {
                let first = body.first().copied().unwrap_or(0);
                // VP6's dimension-adjustment byte is the codec config libavcodec
                // wants as extradata, so it rides the same side channel as the
                // avcC record rather than the payload.
                if matches!(unit.codec, FlvCodec::Vp6 | FlvCodec::Vp6Alpha)
                    && self.video_config.is_none()
                    && video_config.is_none()
                {
                    video_config = body.get(1..2).map(<[u8]>::to_vec);
                }
                if audio_params.is_none() && self.audio_params.is_none() {
                    audio_params = audio_tag_params(unit.codec, first);
                }
                units.push(unit);
            }
            pos = body_end;
        }
        self.buf.drain(..pos);
        self.units.append(&mut units);
        if self.metadata.is_none() {
            self.metadata = metadata;
        }
        if let Some(c) = video_config {
            self.video_config = Some(c);
        }
        if let Some(c) = audio_config {
            self.audio_config = Some(c);
        }
        if self.audio_params.is_none() {
            self.audio_params = audio_params;
        }
    }
}

/// Append a 3-byte big-endian integer (the FLV size / timestamp width).
fn write_u24(out: &mut Vec<u8>, v: u32) {
    out.push((v >> 16) as u8);
    out.push((v >> 8) as u8);
    out.push(v as u8);
}

/// Incremental FLV muxer, the inverse of [`FlvDemuxer`]: wrap each access unit of
/// an elementary stream into an FLV tag. The "FLV" header is written ahead of the
/// first tag; thereafter each tag is preceded by the previous tag's size, matching
/// the layout [`FlvDemuxer`] reads (so a mux -> demux round trip recovers the
/// access units).
///
/// Single-track ([`new`](Self::new)) writes the one track's decoder config once
/// set ([`set_video_config`](Self::set_video_config) /
/// [`set_audio_config`](Self::set_audio_config), M662). The A/V case
/// ([`new_av`](Self::new_av), driven by [`crate::flvmuxn::FlvMuxN`]) carries a
/// video + audio track and writes each track's decoder config (the `avcC`
/// record / AAC `AudioSpecificConfig`) as an FLV sequence-header tag up front,
/// so a player can decode; video and audio media tags then interleave by
/// timestamp.
#[derive(Debug)]
pub struct FlvMuxer {
    tags: TagList,
    /// The video track's codec, `None` for an audio-only muxer.
    video: Option<FlvCodec>,
    /// The audio track's codec, `None` for a video-only muxer.
    audio: Option<FlvCodec>,
    /// H.264 `avcC` decoder configuration record, or VP6's one-byte dimension
    /// adjustment. Empty writes no video sequence-header tag (the single-track /
    /// media-only profile) and a zero VP6 adjustment.
    video_config: Vec<u8>,
    /// AAC `AudioSpecificConfig`; empty writes no audio sequence-header tag.
    audio_config: Vec<u8>,
    /// First byte of every audio tag: sound format + rate / size / channel flags.
    audio_flags: u8,
    header_written: bool,
    prev_tag_size: u32,
}

impl FlvMuxer {
    /// A single-track muxer for one codec (media frames only, no sequence
    /// header). The audio flags default to the codec's usual layout: use
    /// [`set_audio_params`](Self::set_audio_params) for a real MP3 rate.
    pub fn new(codec: FlvCodec) -> Self {
        let video = (codec.track() == FlvTrack::Video).then_some(codec);
        let audio = (codec.track() == FlvTrack::Audio).then_some(codec);
        Self {
            tags: TagList::new(),
            video,
            audio,
            video_config: Vec::new(),
            audio_config: Vec::new(),
            audio_flags: audio_tag_flags(codec, 2, 44_100),
            header_written: false,
            prev_tag_size: 0,
        }
    }

    /// An A/V muxer carrying a video + audio track, each with its decoder config
    /// (the H.264 `avcC` record and AAC `AudioSpecificConfig`) written as an FLV
    /// sequence-header tag ahead of the media frames. A codec with no sequence
    /// header (Sorenson, VP6, MP3, Speex) passes an empty config.
    pub fn new_av(
        video: FlvCodec,
        video_config: Vec<u8>,
        audio: FlvCodec,
        audio_config: Vec<u8>,
    ) -> Self {
        Self {
            tags: TagList::new(),
            video: Some(video),
            audio: Some(audio),
            video_config,
            audio_config,
            audio_flags: audio_tag_flags(audio, 2, 44_100),
            header_written: false,
            prev_tag_size: 0,
        }
    }

    /// Declare the audio track's channel count and sample rate in the tag flags.
    /// Only MP3 varies (AAC and Speex have a fixed flag encoding), and only
    /// before the first tag is emitted.
    pub fn set_audio_params(&mut self, channels: u8, sample_rate: u32) {
        if let Some(codec) = self.audio {
            self.audio_flags = audio_tag_flags(codec, channels, sample_rate);
        }
    }

    /// Attach stream metadata, written as an `onMetaData` script tag ahead of the
    /// first media tag (the inverse of [`FlvDemuxer::metadata`]).
    pub fn with_tags(mut self, tags: TagList) -> Self {
        self.tags = tags;
        self
    }

    /// Set the video decoder config: the `avcC` record written as the video
    /// sequence-header tag (M662), or VP6's one-byte dimension adjustment
    /// written into every video tag. Effective before the first tag is emitted.
    pub fn set_video_config(&mut self, config: Vec<u8>) {
        self.video_config = config;
    }

    /// Set the audio decoder config (the AAC `AudioSpecificConfig`), written as
    /// the audio sequence-header tag; effective before the first tag (M662).
    pub fn set_audio_config(&mut self, asc: Vec<u8>) {
        self.audio_config = asc;
    }

    /// Wrap one access unit into FLV bytes, routed to the muxer's single track:
    /// the legacy single-track entry point (video frames are flagged keyframes).
    pub fn push_au(&mut self, data: &[u8], pts_ms: u32) -> Vec<u8> {
        if self.video.is_some() {
            self.push_video(data, pts_ms, 0, true)
        } else {
            self.push_audio(data, pts_ms)
        }
    }

    /// The FLV video-tag timing of a pipeline frame: `(dts_ms, cts_ms)`. FLV tag
    /// timestamps are decode time; a producer that leaves `dts_ns` unset (0 with
    /// a nonzero pts) gets the legacy `dts = pts` behavior, so only a real
    /// reordering producer yields a nonzero composition offset.
    pub fn video_tag_timing(timing: &g2g_core::FrameTiming) -> (u32, i32) {
        let dts_ns = if timing.dts_ns == 0 && timing.pts_ns != 0 {
            timing.pts_ns
        } else {
            timing.dts_ns
        };
        let dts_ms = (dts_ns / 1_000_000) as u32;
        let cts_ms = ((timing.pts_ns as i64 - dts_ns as i64) / 1_000_000) as i32;
        (dts_ms, cts_ms)
    }

    /// Wrap one video access unit into the FLV bytes to emit (`keyframe` sets the
    /// FLV frame type), prepending the header + sequence headers on the first
    /// call. `dts_ms` is the tag timestamp (FLV timestamps are decode time) and
    /// `cts_ms` the signed composition-time offset (`pts - dts`), so a B-frame
    /// H.264 stream's presentation times survive the mux (the demuxer's `pts =
    /// dts + cts` inverse); an I/P stream passes 0. H.264 takes an AVCC access
    /// unit, the other codecs their own bitstream frame.
    pub fn push_video(&mut self, au: &[u8], dts_ms: u32, cts_ms: i32, keyframe: bool) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_header(&mut out);
        let codec = self.video.unwrap_or(FlvCodec::H264);
        // frame type (1 keyframe, 2 interframe) | codec id.
        let frame_type = if keyframe { 1u8 } else { 2u8 };
        let mut body = alloc::vec![(frame_type << 4) | codec.tag_id()];
        match codec {
            FlvCodec::H264 => {
                // AVC packet type 1 (NALU), then the two's-complement 24-bit
                // composition-time offset.
                let cts = (cts_ms.clamp(-(1 << 23), (1 << 23) - 1) as u32) & 0x00FF_FFFF;
                body.push(0x01);
                body.extend_from_slice(&[(cts >> 16) as u8, (cts >> 8) as u8, cts as u8]);
            }
            // The dimension-adjustment byte the VP6 decoder takes as extradata.
            FlvCodec::Vp6 | FlvCodec::Vp6Alpha => {
                body.push(self.video_config.first().copied().unwrap_or(0))
            }
            _ => {}
        }
        body.extend_from_slice(au);
        self.emit_tag(&mut out, TAG_VIDEO, dts_ms, &body);
        out
    }

    /// Wrap one audio access unit into the FLV bytes to emit, prepending the
    /// header + sequence headers on the first call. AAC takes a raw (de-ADTS'd)
    /// frame, MP3 and Speex their own bitstream frame.
    pub fn push_audio(&mut self, au: &[u8], pts_ms: u32) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_header(&mut out);
        let mut body = alloc::vec![self.audio_flags];
        // AAC packet type 1 (a raw frame, as opposed to the config).
        if self.audio == Some(FlvCodec::Aac) {
            body.push(0x01);
        }
        body.extend_from_slice(au);
        self.emit_tag(&mut out, TAG_AUDIO, pts_ms, &body);
        out
    }

    /// Write the FLV file header and the leading tags (an `onMetaData` script tag
    /// when tags are set, then each present track's sequence header) on the first
    /// call; a no-op afterwards.
    fn write_header(&mut self, out: &mut Vec<u8>) {
        if self.header_written {
            return;
        }
        out.extend_from_slice(b"FLV");
        out.push(1); // version
                     // Flags: bit 0 video present, bit 2 audio present.
        let mut flags = 0u8;
        if self.video.is_some() {
            flags |= 0x01;
        }
        if self.audio.is_some() {
            flags |= 0x04;
        }
        out.push(flags);
        out.extend_from_slice(&(FLV_HEADER_MIN as u32).to_be_bytes()); // data offset
        self.header_written = true;

        if !self.tags.is_empty() {
            let script = on_metadata_body(&self.tags);
            self.emit_tag(out, TAG_SCRIPT, 0, &script);
        }
        // Sequence headers (AVC packet type 0 / AAC packet type 0) carry the
        // decoder config a player needs before the media frames. Only H.264 and
        // AAC have one: VP6's config byte rides each media tag instead.
        if self.video == Some(FlvCodec::H264) && !self.video_config.is_empty() {
            let mut body = alloc::vec![0x17u8, 0x00, 0x00, 0x00, 0x00];
            body.extend_from_slice(&self.video_config);
            self.emit_tag(out, TAG_VIDEO, 0, &body);
        }
        if self.audio == Some(FlvCodec::Aac) && !self.audio_config.is_empty() {
            let mut body = alloc::vec![self.audio_flags, 0x00];
            body.extend_from_slice(&self.audio_config);
            self.emit_tag(out, TAG_AUDIO, 0, &body);
        }
    }

    /// Append one tag, prefixed by the prior tag's `PreviousTagSize` (0 for the
    /// first tag), and record this tag's size for the next.
    fn emit_tag(&mut self, out: &mut Vec<u8>, tag_type: u8, pts_ms: u32, body: &[u8]) {
        out.extend_from_slice(&self.prev_tag_size.to_be_bytes());
        let tag = flv_tag(tag_type, pts_ms, body);
        self.prev_tag_size = tag.len() as u32;
        out.extend_from_slice(&tag);
    }
}

/// Build one FLV tag: 11-byte header (type, 24-bit size, 24+8-bit timestamp,
/// stream id) then the body.
fn flv_tag(tag_type: u8, pts_ms: u32, body: &[u8]) -> Vec<u8> {
    let mut tag = alloc::vec![tag_type];
    write_u24(&mut tag, body.len() as u32);
    write_u24(&mut tag, pts_ms & 0x00FF_FFFF);
    tag.push((pts_ms >> 24) as u8); // timestamp extension
    write_u24(&mut tag, 0); // stream id
    tag.extend_from_slice(body);
    tag
}

/// Serialize a [`TagList`] as an `onMetaData` script body (AMF0): the event-name
/// string then an ECMA array of `key`/string-value properties. The typed keys
/// write their conventional FLV names so they decode back to the same [`Tag`]
/// variant; anything else uses `Tag::key` / `Tag::value_string`, which keeps
/// [`Tag::Other`]'s stored key and flattens the integer / freeform variants FLV
/// has no form for.
fn on_metadata_body(tags: &TagList) -> Vec<u8> {
    let mut b = Vec::new();
    write_amf0_string(&mut b, "onMetaData");
    b.push(AMF0_ECMA_ARRAY);
    b.extend_from_slice(&(tags.tags().len() as u32).to_be_bytes());
    for t in tags.tags() {
        let (key, value) = (t.key(), t.value_string());
        // an object/array key is a raw (unmarked) length-prefixed string.
        b.extend_from_slice(&(key.len() as u16).to_be_bytes());
        b.extend_from_slice(key.as_bytes());
        write_amf0_string(&mut b, &value);
    }
    b.extend_from_slice(&0u16.to_be_bytes()); // empty key precedes the end marker
    b.push(AMF0_OBJECT_END);
    b
}

/// Write a marker-prefixed AMF0 string value.
fn write_amf0_string(out: &mut Vec<u8>, s: &str) {
    out.push(AMF0_STRING);
    out.extend_from_slice(&(s.len() as u16).to_be_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// A sequence-header tag's decoder config, or `None` for any other tag: the
/// `AVCDecoderConfigurationRecord` of an AVC packet-type-0 video tag, or the
/// `AudioSpecificConfig` of an AAC packet-type-0 audio tag (M662).
fn sequence_header(tag_type: u8, body: &[u8]) -> Option<(FlvTrack, &[u8])> {
    match tag_type {
        TAG_VIDEO if *body.first()? & 0x0F == VIDEO_CODEC_AVC && *body.get(1)? == 0 => {
            Some((FlvTrack::Video, body.get(5..)?))
        }
        TAG_AUDIO if *body.first()? >> 4 == SOUND_FORMAT_AAC && *body.get(1)? == 0 => {
            Some((FlvTrack::Audio, body.get(2..)?))
        }
        _ => None,
    }
}

/// Map one FLV tag to an access unit, or `None` for a tag this parser skips
/// (a sequence header, an unsupported codec, or a script/metadata tag).
fn parse_tag(tag_type: u8, timestamp: u32, body: &[u8]) -> Option<FlvUnit> {
    let first = *body.first()?;
    let (codec, keyframe) = match tag_type {
        // body[0] = frame type (high nibble) | codec id (low nibble). Frame type
        // 1 is a keyframe (2 interframe, 3..5 disposable/generated).
        TAG_VIDEO => (FlvCodec::from_video_id(first & 0x0F)?, first >> 4 == 1),
        // body[0] = sound format (high nibble) | rate/size/type (low nibble).
        // Every audio frame is a resync point.
        TAG_AUDIO => (FlvCodec::from_sound_format(first >> 4)?, true),
        _ => return None,
    };
    // H.264 and AAC prefix the payload with a packet type: 0 is the decoder
    // config, kept as the side channel by `sequence_header`, 2 an end-of-sequence
    // marker, and only 1 is a media frame.
    if matches!(codec, FlvCodec::H264 | FlvCodec::Aac) && *body.get(1)? != 1 {
        return None;
    }
    // The tag timestamp is the decode time; H.264's signed 24-bit composition
    // offset gives the presentation time (negative clamps to 0). No other FLV
    // video codec reorders, so their pts is the tag timestamp.
    let pts_ms = if codec == FlvCodec::H264 {
        let cts =
            ((*body.get(2)? as i32) << 16 | (*body.get(3)? as i32) << 8 | *body.get(4)? as i32)
                << 8
                >> 8;
        (timestamp as i64 + cts as i64).clamp(0, u32::MAX as i64) as u32
    } else {
        timestamp
    };
    Some(FlvUnit {
        codec,
        data: body.get(1 + payload_offset(codec)..)?.to_vec(),
        pts_ms,
        dts_ms: timestamp,
        keyframe,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use g2g_core::Tag;

    /// Append a 3-byte big-endian length.
    fn push_u24(out: &mut Vec<u8>, v: u32) {
        out.push((v >> 16) as u8);
        out.push((v >> 8) as u8);
        out.push(v as u8);
    }

    /// Build one FLV tag (without its leading `PreviousTagSize`).
    fn tag(tag_type: u8, timestamp: u32, body: &[u8]) -> Vec<u8> {
        let mut t = vec![tag_type];
        push_u24(&mut t, body.len() as u32);
        push_u24(&mut t, timestamp & 0x00FF_FFFF);
        t.push((timestamp >> 24) as u8);
        push_u24(&mut t, 0); // stream id
        t.extend_from_slice(body);
        t
    }

    /// A video tag body carrying one AVCC access unit (`avc_packet_type` 1).
    fn avc_nalu(au: &[u8]) -> Vec<u8> {
        let mut b = vec![0x17, 0x01, 0x00, 0x00, 0x00]; // keyframe|AVC, NALU, cts=0
        b.extend_from_slice(au);
        b
    }

    /// An audio tag body carrying one raw AAC frame (`aac_packet_type` 1).
    fn aac_raw(frame: &[u8]) -> Vec<u8> {
        let mut b = vec![0xAF, 0x01]; // AAC|44k|16bit|stereo, raw frame
        b.extend_from_slice(frame);
        b
    }

    /// Assemble a full FLV stream from a sequence of tags, including the header
    /// and the `PreviousTagSize` prefixes.
    fn flv_stream(tags: &[Vec<u8>]) -> Vec<u8> {
        let mut s = b"FLV".to_vec();
        s.push(1); // version
        s.push(0x05); // flags: audio + video present
        s.extend_from_slice(&9u32.to_be_bytes()); // data offset
        let mut prev = 0u32;
        for t in tags {
            s.extend_from_slice(&prev.to_be_bytes());
            s.extend_from_slice(t);
            prev = t.len() as u32;
        }
        s
    }

    // regression: an AVC video tag whose body is shorter than the 5-byte AVC
    // header (packet type + 24-bit composition offset) must be skipped, not
    // panic on an out-of-bounds index into the composition offset. Found by
    // fuzzing the demuxer.
    #[test]
    fn truncated_avc_video_tag_does_not_panic() {
        // AVC (codec 7), avc_packet_type 1, then a body that runs out before the
        // 3-byte composition offset is complete
        for body in [
            vec![0x17u8, 0x01],
            vec![0x17, 0x01, 0x00],
            vec![0x17, 0x01, 0x00, 0x00],
        ] {
            let stream = flv_stream(&[tag(TAG_VIDEO, 0, &body)]);
            let mut d = FlvDemuxer::new();
            d.push_data(&stream); // must not panic
            assert!(d.take_units().is_empty(), "truncated tag is skipped");
        }
    }

    #[test]
    fn demuxes_interleaved_video_and_audio() {
        let v0 = [0u8, 0, 0, 5, 0x65, 0x11];
        let a0 = [0x21u8, 0x33];
        let v1 = [0u8, 0, 0, 5, 0x41, 0x22];
        let stream = flv_stream(&[
            tag(TAG_VIDEO, 0, &avc_nalu(&v0)),
            tag(TAG_AUDIO, 0, &aac_raw(&a0)),
            tag(TAG_VIDEO, 33, &avc_nalu(&v1)),
        ]);

        let mut d = FlvDemuxer::new();
        d.push_data(&stream);
        let units = d.take_units();

        assert_eq!(units.len(), 3);
        assert_eq!(
            units[0],
            FlvUnit {
                codec: FlvCodec::H264,
                data: v0.to_vec(),
                pts_ms: 0,
                dts_ms: 0,
                keyframe: true
            }
        );
        assert_eq!(
            units[1],
            FlvUnit {
                codec: FlvCodec::Aac,
                data: a0.to_vec(),
                pts_ms: 0,
                dts_ms: 0,
                keyframe: true
            }
        );
        assert_eq!(
            units[2],
            FlvUnit {
                codec: FlvCodec::H264,
                data: v1.to_vec(),
                pts_ms: 33,
                dts_ms: 33,
                keyframe: true
            }
        );
    }

    #[test]
    fn skips_sequence_headers_and_script_tags() {
        // A video config record (avc_packet_type 0), an onMetaData script tag, and
        // an AAC AudioSpecificConfig (aac_packet_type 0): none are access units.
        let video_config = vec![0x17u8, 0x00, 0, 0, 0, 0x01, 0x64];
        let aac_config = vec![0xAFu8, 0x00, 0x12, 0x10];
        let stream = flv_stream(&[
            tag(TAG_VIDEO, 0, &video_config),
            tag(18, 0, b"onMetaData stuff"),
            tag(TAG_AUDIO, 0, &aac_config),
            tag(TAG_VIDEO, 0, &avc_nalu(&[0x65, 0xAA])),
        ]);

        let mut d = FlvDemuxer::new();
        d.push_data(&stream);
        let units = d.take_units();

        assert_eq!(units.len(), 1, "only the media frame, not the headers");
        assert_eq!(units[0].data, vec![0x65, 0xAA]);
    }

    #[test]
    fn captures_script_metadata_body() {
        let stream = flv_stream(&[
            tag(18, 0, b"onMetaData-amf0-blob"),
            tag(TAG_VIDEO, 0, &avc_nalu(&[0x65, 0xAA])),
        ]);
        let mut d = FlvDemuxer::new();
        d.push_data(&stream);
        assert_eq!(
            d.metadata(),
            Some(&b"onMetaData-amf0-blob"[..]),
            "script body retained"
        );
        assert_eq!(d.take_units().len(), 1, "the media frame still demuxes");
    }

    #[test]
    fn mux_writes_on_metadata_script_tag() {
        let tags: TagList = [Tag::Title("Clip".into()), Tag::Encoder("g2g".into())]
            .into_iter()
            .collect();
        let mut mux = FlvMuxer::new(FlvCodec::H264).with_tags(tags);
        let bytes = mux.push_au(&[0x65, 0xAA], 0);

        // The demuxer retains the first script tag's body; it round-trips to the
        // same tags, and the media AU still demuxes.
        let mut d = FlvDemuxer::new();
        d.push_data(&bytes);
        let meta = d.metadata().expect("script tag body retained");
        assert!(
            meta.starts_with(&[AMF0_STRING, 0, 10]),
            "begins with the onMetaData string"
        );
        assert!(meta.windows(10).any(|w| w == b"onMetaData"));
        assert!(meta.windows(3).any(|w| w == b"g2g"));
        assert_eq!(
            d.take_units().len(),
            1,
            "the media AU still demuxes after the script tag"
        );
    }

    #[test]
    fn mux_without_tags_writes_no_script_tag() {
        let mut mux = FlvMuxer::new(FlvCodec::H264);
        let bytes = mux.push_au(&[0x65, 0xAA], 0);
        let mut d = FlvDemuxer::new();
        d.push_data(&bytes);
        assert!(
            d.metadata().is_none(),
            "no script tag without attached tags"
        );
        assert_eq!(d.take_units().len(), 1);
    }

    #[test]
    fn reassembles_across_chunk_boundaries() {
        let stream = flv_stream(&[
            tag(TAG_VIDEO, 0, &avc_nalu(&[0x65, 0x11, 0x22])),
            tag(TAG_AUDIO, 10, &aac_raw(&[0x33, 0x44])),
        ]);

        // Feed the stream one byte at a time: tags emerge only once whole.
        let mut d = FlvDemuxer::new();
        for &b in &stream {
            d.push_data(&[b]);
        }
        let units = d.take_units();

        assert_eq!(units.len(), 2);
        assert_eq!(units[0].data, vec![0x65, 0x11, 0x22]);
        assert_eq!(units[1].track(), FlvTrack::Audio);
        assert_eq!(units[1].pts_ms, 10);
    }

    #[test]
    fn mux_round_trips_through_demuxer() {
        // The muxer's FLV bytes feed straight back through the demuxer, recovering
        // the access units, their order, and their timestamps.
        let aus: [&[u8]; 2] = [&[0x65, 0xAA, 0xBB], &[0x41, 0xCC]];
        let mut mux = FlvMuxer::new(FlvCodec::H264);
        let mut stream = Vec::new();
        stream.extend_from_slice(&mux.push_au(aus[0], 0));
        stream.extend_from_slice(&mux.push_au(aus[1], 33));

        let mut demux = FlvDemuxer::new();
        demux.push_data(&stream);
        let units = demux.take_units();
        assert_eq!(
            units,
            vec![
                FlvUnit {
                    codec: FlvCodec::H264,
                    data: aus[0].to_vec(),
                    pts_ms: 0,
                    dts_ms: 0,
                    keyframe: true
                },
                FlvUnit {
                    codec: FlvCodec::H264,
                    data: aus[1].to_vec(),
                    pts_ms: 33,
                    dts_ms: 33,
                    keyframe: true
                },
            ]
        );
    }

    #[test]
    fn mux_writes_composition_offset() {
        // A reordered (B-frame) stream: dts 100 with pts 133 writes cts +33,
        // and the demuxer's `pts = dts + cts` inverse recovers both times.
        let mut mux = FlvMuxer::new(FlvCodec::H264);
        let mut stream = mux.push_video(&[0x65, 0xAA], 100, 33, true);
        stream.extend_from_slice(&mux.push_video(&[0x41, 0xBB], 133, 0, false));

        let mut d = FlvDemuxer::new();
        d.push_data(&stream);
        let units = d.take_units();
        assert_eq!((units[0].dts_ms, units[0].pts_ms), (100, 133), "cts +33");
        assert_eq!((units[1].dts_ms, units[1].pts_ms), (133, 133), "cts 0");

        // The pipeline timing mapping: an unset dts (0) keeps pts == dts; a
        // real dts yields the offset.
        use g2g_core::FrameTiming;
        let legacy = FrameTiming {
            pts_ns: 133_000_000,
            ..Default::default()
        };
        assert_eq!(FlvMuxer::video_tag_timing(&legacy), (133, 0));
        let reordered = FrameTiming {
            pts_ns: 133_000_000,
            dts_ns: 100_000_000,
            ..Default::default()
        };
        assert_eq!(FlvMuxer::video_tag_timing(&reordered), (100, 33));
    }

    #[test]
    fn mux_writes_audio_tags() {
        let mut mux = FlvMuxer::new(FlvCodec::Aac);
        let bytes = mux.push_au(&[0x11, 0x22], 10);
        // "FLV" header, then a demuxer recovers the AAC frame.
        assert_eq!(&bytes[0..3], b"FLV");
        let mut demux = FlvDemuxer::new();
        demux.push_data(&bytes);
        let units = demux.take_units();
        assert_eq!(
            units,
            vec![FlvUnit {
                codec: FlvCodec::Aac,
                data: vec![0x11, 0x22],
                pts_ms: 10,
                dts_ms: 10,
                keyframe: true
            }]
        );
    }

    #[test]
    fn captures_sequence_header_configs() {
        // avcC record body after the 5-byte AVC tag prefix; ASC after the 2-byte
        // AAC tag prefix.
        let avcc = [0x01u8, 0x64, 0x00, 0x1E, 0xFF, 0xE1];
        let asc = [0x12u8, 0x10];
        let mut video_body = vec![0x17u8, 0x00, 0, 0, 0];
        video_body.extend_from_slice(&avcc);
        let mut audio_body = vec![0xAFu8, 0x00];
        audio_body.extend_from_slice(&asc);
        let stream = flv_stream(&[
            tag(TAG_VIDEO, 0, &video_body),
            tag(TAG_AUDIO, 0, &audio_body),
            tag(TAG_VIDEO, 0, &avc_nalu(&[0x65, 0xAA])),
        ]);

        let mut d = FlvDemuxer::new();
        d.push_data(&stream);
        assert_eq!(
            d.video_config(),
            Some(&avcc[..]),
            "avcC retained as the side channel"
        );
        assert_eq!(
            d.audio_config(),
            Some(&asc[..]),
            "AudioSpecificConfig retained"
        );
        assert_eq!(d.take_units().len(), 1, "config tags are not media units");
    }

    #[test]
    fn video_cts_offsets_pts_from_dts() {
        // A video tag at dts 100 with composition offset +33, and one with a
        // negative offset that clamps at 0.
        let mut plus = vec![0x27u8, 0x01, 0x00, 0x00, 0x21]; // interframe, cts +33
        plus.extend_from_slice(&[0xAA]);
        let mut neg = vec![0x27u8, 0x01, 0xFF, 0xFF, 0xD6]; // cts -42
        neg.extend_from_slice(&[0xBB]);
        let stream = flv_stream(&[tag(TAG_VIDEO, 100, &plus), tag(TAG_VIDEO, 10, &neg)]);

        let mut d = FlvDemuxer::new();
        d.push_data(&stream);
        let units = d.take_units();
        assert_eq!(
            (units[0].dts_ms, units[0].pts_ms),
            (100, 133),
            "pts = dts + cts"
        );
        assert!(!units[0].keyframe, "frame type 2 is an interframe");
        assert_eq!(
            (units[1].dts_ms, units[1].pts_ms),
            (10, 0),
            "negative pts clamps to 0"
        );
    }

    #[test]
    fn ignores_codecs_outside_the_carried_set() {
        // A Nellymoser audio tag (sound format 6) and a screen-video tag (codec
        // id 3): neither is carried, so neither yields a unit.
        let nellymoser = vec![0x62u8, 0xAA, 0xBB];
        let screen = vec![0x13u8, 0xCC, 0xDD];
        let stream = flv_stream(&[tag(TAG_AUDIO, 0, &nellymoser), tag(TAG_VIDEO, 0, &screen)]);

        let mut d = FlvDemuxer::new();
        d.push_data(&stream);
        assert!(d.take_units().is_empty());
    }

    #[test]
    fn demuxes_the_legacy_codecs_at_their_payload_offsets() {
        // Sorenson (id 2) and MP3 (format 2) start their payload right after the
        // first byte; VP6 (id 4) and VP6-alpha (id 5) spend one more on the
        // dimension adjustment, and Speex (format 11) none.
        let sorenson = vec![0x12u8, 0xAA, 0xBB];
        let vp6 = vec![0x14u8, 0x22, 0xCC, 0xDD];
        let vp6a = vec![0x25u8, 0x22, 0x00, 0x00, 0x02, 0xEE];
        let mp3 = vec![0x2Fu8, 0x11, 0x22]; // 44.1 kHz stereo
        let speex = vec![0xB6u8, 0x33];
        let stream = flv_stream(&[
            tag(TAG_VIDEO, 0, &sorenson),
            tag(TAG_VIDEO, 10, &vp6),
            tag(TAG_VIDEO, 20, &vp6a),
            tag(TAG_AUDIO, 0, &mp3),
            tag(TAG_AUDIO, 10, &speex),
        ]);

        let mut d = FlvDemuxer::new();
        d.push_data(&stream);
        let units = d.take_units();
        let got: Vec<_> = units.iter().map(|u| (u.codec, u.data.clone())).collect();
        assert_eq!(
            got,
            vec![
                (FlvCodec::SorensonH263, vec![0xAA, 0xBB]),
                (FlvCodec::Vp6, vec![0xCC, 0xDD]),
                // The 24-bit alpha offset stays in the payload: its decoder reads it.
                (FlvCodec::Vp6Alpha, vec![0x00, 0x00, 0x02, 0xEE]),
                (FlvCodec::Mp3, vec![0x11, 0x22]),
                (FlvCodec::Speex, vec![0x33]),
            ]
        );
        assert!(units[0].keyframe, "frame type 1 is a keyframe");
        assert!(!units[2].keyframe, "frame type 2 is an interframe");
        assert_eq!(units[3].track(), FlvTrack::Audio);
        assert_eq!(
            d.video_config(),
            Some(&[0x22u8][..]),
            "the VP6 adjustment byte is the video config side channel"
        );
        assert_eq!(
            d.audio_params(),
            Some((2, 44_100)),
            "the MP3 tag flags declare the layout"
        );
    }

    #[test]
    fn mp3_at_8khz_uses_sound_format_14() {
        // Sound format 14 is MP3 whose rate the 2-bit field cannot express.
        let stream = flv_stream(&[tag(TAG_AUDIO, 0, &[0xE2u8, 0x77])]);
        let mut d = FlvDemuxer::new();
        d.push_data(&stream);
        let units = d.take_units();
        assert_eq!(units[0].codec, FlvCodec::Mp3);
        assert_eq!(units[0].data, vec![0x77]);
        assert_eq!(d.audio_params(), Some((1, 8_000)));
    }

    #[test]
    fn legacy_codecs_round_trip_through_the_muxer() {
        // Each codec's mux -> demux recovers the payload byte for byte, and the
        // VP6 adjustment travels via the config side channel.
        for (codec, config) in [
            (FlvCodec::SorensonH263, Vec::new()),
            (FlvCodec::Vp6, vec![0x22u8]),
            (FlvCodec::Vp6Alpha, vec![0x22u8]),
        ] {
            let mut mux = FlvMuxer::new(codec);
            mux.set_video_config(config.clone());
            let mut stream = mux.push_video(&[0xAA, 0xBB], 0, 0, true);
            stream.extend_from_slice(&mux.push_video(&[0xCC], 40, 0, false));

            let mut d = FlvDemuxer::new();
            d.push_data(&stream);
            let units = d.take_units();
            assert_eq!(units.len(), 2, "{codec:?}");
            assert_eq!(units[0].codec, codec);
            assert_eq!(units[0].data, vec![0xAA, 0xBB], "{codec:?}");
            assert!(units[0].keyframe);
            assert_eq!((units[1].data.clone(), units[1].dts_ms), (vec![0xCC], 40));
            assert!(!units[1].keyframe);
            assert_eq!(d.video_config().unwrap_or_default(), &config[..]);
        }

        for (codec, channels, rate) in [
            (FlvCodec::Mp3, 2, 44_100),
            (FlvCodec::Mp3, 1, 22_050),
            (FlvCodec::Speex, 1, 16_000),
        ] {
            let mut mux = FlvMuxer::new(codec);
            mux.set_audio_params(channels, rate);
            let stream = mux.push_audio(&[0x11, 0x22], 10);

            let mut d = FlvDemuxer::new();
            d.push_data(&stream);
            let units = d.take_units();
            assert_eq!(units[0].codec, codec);
            assert_eq!(units[0].data, vec![0x11, 0x22], "{codec:?}");
            assert_eq!(units[0].pts_ms, 10);
            assert_eq!(d.audio_params(), Some((channels, rate)), "{codec:?}");
        }
    }
}
