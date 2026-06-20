//! Pure FLV (Flash Video) container parser (M119), the byte-stream sibling of
//! [`crate::mpegts::TsDemuxer`] / [`crate::ogg::OggDemuxer`]. `no_std`, no I/O.
//!
//! FLV is a flat tag stream: a 9-byte header, then `PreviousTagSize` (UI32) /
//! tag pairs. Each tag is an 11-byte header (type, 24-bit data size, 24+8-bit
//! millisecond timestamp, stream id) followed by its body. The body's first byte
//! identifies the codec; for the two modern RTMP/FLV codecs this parser forwards
//! the elementary access units: H.264 (video codec id 7, AVCC length-prefixed
//! NALUs) and AAC (audio sound format 10, raw frames).
//!
//! Scope (v1): H.264 video + AAC audio media frames. The sequence-header tags
//! (the `AVCDecoderConfigurationRecord` / `AudioSpecificConfig`) are skipped, the
//! codec-config / extradata side channel being a shared demuxer follow-up. The
//! `onMetaData` script tag's body is retained ([`FlvDemuxer::metadata`]) so the
//! element can surface its AMF0 metadata via the tag system. Other codecs (VP6,
//! H.263, MP3, Speex) are ignored.

use alloc::vec::Vec;

use g2g_core::{Tag, TagList};

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

/// One demuxed access unit: the elementary stream it belongs to, its payload
/// (AVCC NALUs for H.264, a raw AAC frame for audio), and its millisecond
/// presentation timestamp from the tag header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlvUnit {
    pub track: FlvTrack,
    pub data: Vec<u8>,
    pub pts_ms: u32,
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
            if tag_type == TAG_SCRIPT && metadata.is_none() {
                metadata = Some(self.buf[body_start..body_end].to_vec());
            } else if let Some(unit) = parse_tag(tag_type, timestamp, &self.buf[body_start..body_end]) {
                units.push(unit);
            }
            pos = body_end;
        }
        self.buf.drain(..pos);
        self.units.append(&mut units);
        if self.metadata.is_none() {
            self.metadata = metadata;
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
/// one elementary stream into an FLV tag. The "FLV" header is written ahead of the
/// first tag; thereafter each tag is preceded by the previous tag's size, matching
/// the layout [`FlvDemuxer`] reads (so a mux -> demux round trip recovers the
/// access units). v1 writes media frames only (no sequence header), mirroring the
/// demuxer's scope.
#[derive(Debug)]
pub struct FlvMuxer {
    track: FlvTrack,
    tags: TagList,
    header_written: bool,
    prev_tag_size: u32,
}

impl FlvMuxer {
    pub fn new(track: FlvTrack) -> Self {
        Self { track, tags: TagList::new(), header_written: false, prev_tag_size: 0 }
    }

    /// Attach stream metadata, written as an `onMetaData` script tag ahead of the
    /// first media tag (the inverse of [`FlvDemuxer::metadata`]).
    pub fn with_tags(mut self, tags: TagList) -> Self {
        self.tags = tags;
        self
    }

    /// Wrap one access unit (an AVCC unit for video, a raw AAC frame for audio)
    /// into the FLV bytes to emit, prepending the file header (+ an `onMetaData`
    /// script tag when tags are set) on the first call.
    pub fn push_au(&mut self, data: &[u8], pts_ms: u32) -> Vec<u8> {
        let mut out = Vec::new();
        if !self.header_written {
            out.extend_from_slice(b"FLV");
            out.push(1); // version
            // Flags: bit 0 video present, bit 2 audio present.
            out.push(match self.track {
                FlvTrack::Video => 0x01,
                FlvTrack::Audio => 0x04,
            });
            out.extend_from_slice(&(FLV_HEADER_MIN as u32).to_be_bytes()); // data offset
            if !self.tags.is_empty() {
                // The script tag is the first tag, so its PreviousTagSize is 0.
                out.extend_from_slice(&0u32.to_be_bytes());
                let script = flv_tag(TAG_SCRIPT, 0, &on_metadata_body(&self.tags));
                self.prev_tag_size = script.len() as u32;
                out.extend_from_slice(&script);
            }
            self.header_written = true;
        }
        // PreviousTagSize: 0 (or the script tag's length) before the first media
        // tag, then the prior tag's length.
        out.extend_from_slice(&self.prev_tag_size.to_be_bytes());
        let tag = self.build_tag(data, pts_ms);
        self.prev_tag_size = tag.len() as u32;
        out.extend_from_slice(&tag);
        out
    }

    /// Build one media tag (11-byte header + codec-tagged body) for an access unit.
    fn build_tag(&self, data: &[u8], pts_ms: u32) -> Vec<u8> {
        let (tag_type, mut body) = match self.track {
            // keyframe | AVC, NALU packet, composition time 0.
            FlvTrack::Video => (TAG_VIDEO, alloc::vec![0x17u8, 0x01, 0x00, 0x00, 0x00]),
            // AAC | 44k | 16-bit | stereo, raw frame.
            FlvTrack::Audio => (TAG_AUDIO, alloc::vec![0xAFu8, 0x01]),
        };
        body.extend_from_slice(data);
        flv_tag(tag_type, pts_ms, &body)
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
/// variant; [`Tag::Other`] keeps its stored key.
fn on_metadata_body(tags: &TagList) -> Vec<u8> {
    let mut b = Vec::new();
    write_amf0_string(&mut b, "onMetaData");
    b.push(AMF0_ECMA_ARRAY);
    b.extend_from_slice(&(tags.tags().len() as u32).to_be_bytes());
    for t in tags.tags() {
        let (key, value) = tag_key_value(t);
        // an object/array key is a raw (unmarked) length-prefixed string.
        b.extend_from_slice(&(key.len() as u16).to_be_bytes());
        b.extend_from_slice(key.as_bytes());
        write_amf0_string(&mut b, value);
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

/// A tag's FLV `onMetaData` key / value. Typed keys use the conventional
/// lowercase names so they round-trip through `Tag::from_key_value`;
/// [`Tag::Other`] keeps its stored key.
fn tag_key_value(tag: &Tag) -> (&str, &str) {
    match tag {
        Tag::Title(v) => ("title", v),
        Tag::Artist(v) => ("artist", v),
        Tag::Album(v) => ("album", v),
        Tag::Encoder(v) => ("encoder", v),
        Tag::Language(v) => ("language", v),
        Tag::Comment(v) => ("comment", v),
        Tag::Other { key, value } => (key, value),
    }
}

/// Map one FLV tag to an access unit, or `None` for a tag this parser skips
/// (a sequence header, an unsupported codec, or a script/metadata tag).
fn parse_tag(tag_type: u8, timestamp: u32, body: &[u8]) -> Option<FlvUnit> {
    match tag_type {
        TAG_VIDEO => {
            // body[0] = frame type (high nibble) | codec id (low nibble).
            let codec_id = body.first()? & 0x0F;
            if codec_id != VIDEO_CODEC_AVC {
                return None;
            }
            // AVC: body[1] = packet type (0 config, 1 NALU, 2 end), body[2..5] =
            // composition time offset, body[5..] = the AVCC access unit.
            if *body.get(1)? != 1 {
                return None;
            }
            Some(FlvUnit { track: FlvTrack::Video, data: body.get(5..)?.to_vec(), pts_ms: timestamp })
        }
        TAG_AUDIO => {
            // body[0] = sound format (high nibble) | rate/size/type (low nibble).
            let sound_format = body.first()? >> 4;
            if sound_format != SOUND_FORMAT_AAC {
                return None;
            }
            // AAC: body[1] = packet type (0 AudioSpecificConfig, 1 raw frame),
            // body[2..] = the raw AAC frame.
            if *body.get(1)? != 1 {
                return None;
            }
            Some(FlvUnit { track: FlvTrack::Audio, data: body.get(2..)?.to_vec(), pts_ms: timestamp })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

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
        assert_eq!(units[0], FlvUnit { track: FlvTrack::Video, data: v0.to_vec(), pts_ms: 0 });
        assert_eq!(units[1], FlvUnit { track: FlvTrack::Audio, data: a0.to_vec(), pts_ms: 0 });
        assert_eq!(units[2], FlvUnit { track: FlvTrack::Video, data: v1.to_vec(), pts_ms: 33 });
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
        assert_eq!(d.metadata(), Some(&b"onMetaData-amf0-blob"[..]), "script body retained");
        assert_eq!(d.take_units().len(), 1, "the media frame still demuxes");
    }

    #[test]
    fn mux_writes_on_metadata_script_tag() {
        let tags: TagList =
            [Tag::Title("Clip".into()), Tag::Encoder("g2g".into())].into_iter().collect();
        let mut mux = FlvMuxer::new(FlvTrack::Video).with_tags(tags);
        let bytes = mux.push_au(&[0x65, 0xAA], 0);

        // The demuxer retains the first script tag's body; it round-trips to the
        // same tags, and the media AU still demuxes.
        let mut d = FlvDemuxer::new();
        d.push_data(&bytes);
        let meta = d.metadata().expect("script tag body retained");
        assert!(meta.starts_with(&[AMF0_STRING, 0, 10]), "begins with the onMetaData string");
        assert!(meta.windows(10).any(|w| w == b"onMetaData"));
        assert!(meta.windows(3).any(|w| w == b"g2g"));
        assert_eq!(d.take_units().len(), 1, "the media AU still demuxes after the script tag");
    }

    #[test]
    fn mux_without_tags_writes_no_script_tag() {
        let mut mux = FlvMuxer::new(FlvTrack::Video);
        let bytes = mux.push_au(&[0x65, 0xAA], 0);
        let mut d = FlvDemuxer::new();
        d.push_data(&bytes);
        assert!(d.metadata().is_none(), "no script tag without attached tags");
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
        assert_eq!(units[1].track, FlvTrack::Audio);
        assert_eq!(units[1].pts_ms, 10);
    }

    #[test]
    fn mux_round_trips_through_demuxer() {
        // The muxer's FLV bytes feed straight back through the demuxer, recovering
        // the access units, their order, and their timestamps.
        let aus: [&[u8]; 2] = [&[0x65, 0xAA, 0xBB], &[0x41, 0xCC]];
        let mut mux = FlvMuxer::new(FlvTrack::Video);
        let mut stream = Vec::new();
        stream.extend_from_slice(&mux.push_au(aus[0], 0));
        stream.extend_from_slice(&mux.push_au(aus[1], 33));

        let mut demux = FlvDemuxer::new();
        demux.push_data(&stream);
        let units = demux.take_units();
        assert_eq!(
            units,
            vec![
                FlvUnit { track: FlvTrack::Video, data: aus[0].to_vec(), pts_ms: 0 },
                FlvUnit { track: FlvTrack::Video, data: aus[1].to_vec(), pts_ms: 33 },
            ]
        );
    }

    #[test]
    fn mux_writes_audio_tags() {
        let mut mux = FlvMuxer::new(FlvTrack::Audio);
        let bytes = mux.push_au(&[0x11, 0x22], 10);
        // "FLV" header, then a demuxer recovers the AAC frame.
        assert_eq!(&bytes[0..3], b"FLV");
        let mut demux = FlvDemuxer::new();
        demux.push_data(&bytes);
        let units = demux.take_units();
        assert_eq!(units, vec![FlvUnit { track: FlvTrack::Audio, data: vec![0x11, 0x22], pts_ms: 10 }]);
    }

    #[test]
    fn ignores_non_aac_non_avc_codecs() {
        // An MP3 audio tag (sound format 2) and an H.263 video tag (codec id 2).
        let mp3 = vec![0x2Fu8, 0xAA, 0xBB];
        let h263 = vec![0x12u8, 0xCC, 0xDD];
        let stream = flv_stream(&[tag(TAG_AUDIO, 0, &mp3), tag(TAG_VIDEO, 0, &h263)]);

        let mut d = FlvDemuxer::new();
        d.push_data(&stream);
        assert!(d.take_units().is_empty());
    }
}
