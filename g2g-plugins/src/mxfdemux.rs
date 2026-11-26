// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::ops::Range;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::caps::framerate_q16_from_fraction;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::log::Target;
use g2g_core::runtime::{ElementSeekRoute, SeekController};
use g2g_core::{
    pcm_from_gst_format, pcm_gst_format, AudioFormat, BusHandle, BusMessage, ByteStreamEncoding,
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MultiOutputElement, MultiOutputSink, PadTemplate, PadTemplates, ProResVariant, PropError,
    PropKind, PropValue, PropertySpec, Rate, Stream, StreamCollection, StreamType, VideoCodec,
};

use crate::avidemux::pcm_alternatives;
use crate::demuxseek::{DemuxSeek, TailRead};
use crate::mkvdemux::{prores_stream_from_str, prores_stream_str};
use crate::mxf::{
    d10_pcm, header_partition_metadata, index_segment, key_kind, partition_pack, random_index_pack,
    random_index_pack_len, read_klv, whole_klv, Essence, EssenceTrack, IndexEntry, IndexSegment,
    KeyKind, Klv, KlvError, PartitionPack, PcmPacking, Rational, MAX_D10_SOUND_BYTES,
    QUANTIZATION_FORMATS,
};
use crate::oneport::OnePortElement;
use crate::typefind::container_video_caps;

fn input_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::Mxf,
    }
}

pub(crate) fn track_caps(track: &EssenceTrack) -> Caps {
    match track.essence {
        Essence::Video {
            codec,
            width,
            height,
        } => {
            let edit_rate = track.edit_rate;
            let framerate = framerate_q16_from_fraction(edit_rate.numerator, edit_rate.denominator)
                .map_or(crate::typefind::PLACEHOLDER_FRAMERATE, Rate::Fixed);
            Caps::CompressedVideo {
                codec,
                width: Dim::Fixed(width),
                height: Dim::Fixed(height),
                framerate,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
                pixel_aspect: g2g_core::PixelAspect::Any,
                alignment: crate::annexb::sample_alignment(codec),
            }
        }
        Essence::Pcm {
            format,
            channels,
            sample_rate,
            ..
        } => Caps::Audio {
            format,
            channels,
            sample_rate,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        },
    }
}

fn is_video(track: &EssenceTrack) -> bool {
    matches!(track.essence, Essence::Video { .. })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MxfStream {
    Video(VideoCodec),
    Pcm(AudioFormat),
}

const DEFAULT_STREAM: MxfStream = MxfStream::Video(VideoCodec::H264);

// the `stream=` names of the picture codecs besides prores, which goes by its profile
const VIDEO_STREAM_NAMES: [(&str, VideoCodec); 4] = [
    ("h264", VideoCodec::H264),
    ("mpeg2", VideoCodec::Mpeg2),
    ("jpeg2000", VideoCodec::Jpeg2000),
    ("dnxhd", VideoCodec::DnxHd),
];

impl MxfStream {
    fn of(track: &EssenceTrack) -> Self {
        match track.essence {
            Essence::Video { codec, .. } => MxfStream::Video(codec),
            Essence::Pcm { format, .. } => MxfStream::Pcm(format),
        }
    }

    fn all() -> impl Iterator<Item = MxfStream> {
        VIDEO_STREAM_NAMES
            .into_iter()
            .map(|(_, codec)| codec)
            .chain(
                ProResVariant::ALL
                    .into_iter()
                    .map(|variant| VideoCodec::ProRes { variant }),
            )
            .map(MxfStream::Video)
            .chain(
                QUANTIZATION_FORMATS
                    .into_iter()
                    .map(|(_, format)| MxfStream::Pcm(format)),
            )
    }

    fn name(self) -> &'static str {
        match self {
            MxfStream::Video(VideoCodec::ProRes { variant }) => prores_stream_str(variant),
            MxfStream::Video(codec) => VIDEO_STREAM_NAMES
                .into_iter()
                .find_map(|(name, known)| (known == codec).then_some(name))
                .expect("the mxf header parser maps only the codecs VIDEO_STREAM_NAMES lists"),
            MxfStream::Pcm(format) => pcm_gst_format(format).expect("mxf sound is pcm"),
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        let codec = VIDEO_STREAM_NAMES
            .into_iter()
            .find_map(|(known, codec)| (known == name).then_some(codec))
            .or_else(|| prores_stream_from_str(name).map(|variant| VideoCodec::ProRes { variant }));
        if let Some(codec) = codec {
            return Some(MxfStream::Video(codec));
        }
        let format = pcm_from_gst_format(name)?;
        QUANTIZATION_FORMATS
            .into_iter()
            .any(|(_, known)| known == format)
            .then_some(MxfStream::Pcm(format))
    }

    fn placeholder_caps(self) -> Vec<Caps> {
        match self {
            MxfStream::Video(codec) => Vec::from([container_video_caps(codec)]),
            // a lone rate and channel wildcard cannot fixate
            MxfStream::Pcm(format) => pcm_alternatives(format),
        }
    }

    fn first_placeholder_caps(self) -> Caps {
        self.placeholder_caps()
            .into_iter()
            .next()
            .expect("every stream has a placeholder")
    }
}

#[derive(Debug, Clone, Copy)]
struct TrackSelection {
    stream: MxfStream,
    // None takes the first track of the stream
    track_number: Option<u32>,
}

impl TrackSelection {
    fn picks(self, track: &EssenceTrack) -> bool {
        MxfStream::of(track) == self.stream
            && self
                .track_number
                .is_none_or(|number| number == track.track_number)
    }
}

// the tracks of an mxf file prefix the demuxer forwards, in file order
pub fn forwardable_tracks(prefix: &[u8]) -> Vec<MxfTrackInfo> {
    let Some(metadata) = header_partition_metadata(prefix) else {
        return Vec::new();
    };
    metadata
        .tracks
        .iter()
        .map(|track| MxfTrackInfo {
            track: *track,
            caps: track_caps(track),
            video: is_video(track),
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct MxfTrackInfo {
    track: EssenceTrack,
    pub caps: Caps,
    pub video: bool,
}

impl MxfTrackInfo {
    pub fn stream_name(&self) -> &'static str {
        MxfStream::of(&self.track).name()
    }

    pub fn track_number(&self) -> u32 {
        self.track.track_number
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PartitionPlace {
    at: u64,
    body_sid: u32,
    body_offset: u64,
    // where stream offset `body_offset` lies: the first essence key, or its value for clip-wrapped essence
    essence_at: Option<u64>,
}

#[derive(Debug, Default)]
struct MxfIndex {
    segments: Vec<IndexSegment>,
    partitions: Vec<PartitionPlace>,
}

impl MxfIndex {
    fn add_segment(&mut self, segment: IndexSegment) {
        let known = self
            .segments
            .iter()
            .any(|known| known.body_sid == segment.body_sid && known.start == segment.start);
        if !known {
            self.segments.push(segment);
            self.segments.sort_by_key(|segment| segment.start);
        }
    }

    fn add_partition(&mut self, at: u64, pack: &PartitionPack) {
        if self.partitions.iter().any(|place| place.at == at) {
            return;
        }
        self.partitions.push(PartitionPlace {
            at,
            body_sid: pack.body_sid,
            body_offset: pack.body_offset,
            essence_at: None,
        });
        self.partitions.sort_by_key(|place| place.at);
    }

    // the essence start of the partition at `partition_at`, set once from its first essence klv
    fn place_essence(&mut self, partition_at: u64, essence_at: u64) {
        if let Some(place) = self
            .partitions
            .iter_mut()
            .find(|place| place.at == partition_at && place.essence_at.is_none())
        {
            place.essence_at = Some(essence_at);
        }
    }

    // the file byte holding the edit unit, as ffmpeg's mxf_absolute_bodysid_offset finds it
    fn file_offset(&self, unit: u64) -> Option<u64> {
        let segment = self.segment_of(unit)?;
        let stream_offset = if segment.edit_unit_byte_count > 0 {
            u64::from(segment.edit_unit_byte_count).checked_mul(unit)?
        } else {
            self.entry(unit)?.stream_offset
        };
        let (slot, place) = self
            .partitions
            .iter()
            .enumerate()
            .filter(|(_, place)| {
                place.body_sid != 0
                    && place.body_sid == segment.body_sid
                    && place.body_offset <= stream_offset
            })
            .max_by_key(|(_, place)| place.body_offset)?;
        let offset = place
            .essence_at?
            .checked_add(stream_offset - place.body_offset)?;
        self.partitions
            .get(slot + 1)
            .is_none_or(|next| offset < next.at)
            .then_some(offset)
    }

    fn segment_of(&self, unit: u64) -> Option<&IndexSegment> {
        self.segments.iter().find(|segment| {
            segment.start <= unit && unit < segment.start.saturating_add(segment.duration)
        })
    }

    fn entry(&self, unit: u64) -> Option<&IndexEntry> {
        let segment = self.segment_of(unit)?;
        let at = usize::try_from(unit - segment.start).ok()?;
        segment.entries.get(entry_slot(segment, at)?)
    }

    fn edit_rate(&self) -> Option<Rational> {
        self.segments.first().map(|segment| segment.edit_rate)
    }

    // the stored unit of the keyframe the frame shown at `display` decodes from
    fn keyframe_for(&self, display: u64) -> Option<u64> {
        let entry = self.entry(display)?;
        let stored = display.checked_add_signed(i64::from(entry.temporal_offset))?;
        let key_frame_offset = self.entry(stored)?.key_frame_offset;
        stored.checked_add_signed(i64::from(key_frame_offset))
    }
}

// avid writes two entries per edit unit and one more, the second of each pair being the one ffmpeg reads
fn entry_slot(segment: &IndexSegment, at: usize) -> Option<usize> {
    let duration = usize::try_from(segment.duration).ok()?;
    let avid = segment.entries.len() == duration.checked_mul(2)?.checked_add(1)?;
    match avid {
        true => at.checked_mul(2),
        false => Some(at),
    }
}

#[derive(Debug, Default)]
struct Reordering {
    display_of_stored: BTreeMap<u64, u64>,
    delay: u64,
}

impl Reordering {
    fn of(index: &MxfIndex) -> Self {
        let mut reordering = Reordering::default();
        for segment in &index.segments {
            reordering.extend(segment);
        }
        reordering.delay = reordering
            .display_of_stored
            .iter()
            .map(|(stored, display)| stored.saturating_sub(*display))
            .max()
            .unwrap_or(0);
        reordering
    }

    // a segment read after streaming started keeps the delay the first frames went out with
    fn extend(&mut self, segment: &IndexSegment) {
        for (at, entry) in segment.entries.iter().enumerate() {
            let Some(display) = segment.start.checked_add(at as u64) else {
                break;
            };
            if let Some(stored) = display.checked_add_signed(i64::from(entry.temporal_offset)) {
                self.display_of_stored.insert(stored, display);
            }
        }
    }

    fn display(&self, stored: u64) -> u64 {
        self.display_of_stored
            .get(&stored)
            .copied()
            .unwrap_or(stored)
    }
}

#[derive(Debug)]
struct Unit {
    port: usize,
    data: Vec<u8>,
    timing: FrameTiming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MxfRead {
    // until the header partition's metadata arrives
    Head,
    // the tail is read through a byte seek for the random index pack
    Tail { expanded: bool },
    // read a partition's index segments and locate its first essence key
    Index { next: u64, end: u64 },
    // the seek back to byte 0 is in flight
    Rewind,
    // each klv goes out as it arrives, the next key at `next`
    Klvs { next: u64 },
    // the file is held whole and read at Eos
    Whole,
}

// a klv value sent on in pieces as its bytes arrive: sound, or clip-wrapped pictures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SplitValue {
    port: usize,
    at: u64,
    end: u64,
}

// where an indexed seek lands, and the edit unit or sample the tracks restart at
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Landing {
    offset: u64,
    edit_unit: u64,
    // set when the landing is inside a clip-wrapped value
    split: Option<SplitValue>,
}

// frames past this size fail the read instead of buffering a bogus klv length
const MAX_PICTURE_BYTES: u64 = 256 * 1024 * 1024;
// sound goes out in pieces of at most this many per second, the picture rate of ffmpeg's frame-wrapped sound
const SOUND_PIECES_PER_SECOND: u32 = 25;
// header metadata past this size fails the read
const MAX_HEADER_BYTES: usize = 64 * 1024 * 1024;
const LOG_CATEGORY: &str = "mxfdemux";

fn malformed(reason: &str) -> G2gError {
    g2g_core::g2g_error!(Target::category(LOG_CATEGORY), "{reason}");
    G2gError::CapsMismatch
}

fn sample_block(essence: Essence) -> Option<u64> {
    let Essence::Pcm {
        format, channels, ..
    } = essence
    else {
        return None;
    };
    Some(u64::try_from(format.bytes_per_sample()?).ok()? * u64::from(channels))
}

fn samples_ns(samples: u64, sample_rate: u32) -> u64 {
    u64::try_from(u128::from(samples) * 1_000_000_000 / u128::from(sample_rate)).unwrap_or(u64::MAX)
}

// the byte a seek to `target_ns` restarts at: the keyframe before it, or for sound alone the sample it names
fn landing(
    tracks: &[EssenceTrack],
    index: &MxfIndex,
    clip_value: Option<SplitValue>,
    target_ns: u64,
) -> Option<Landing> {
    let in_clip = |offset: u64| clip_value.filter(|clip| (clip.at..clip.end).contains(&offset));
    if tracks.iter().any(is_video) {
        let rate = index.edit_rate()?;
        let edit_unit = index.keyframe_for(rate.ns_to_units(target_ns))?;
        let offset = index.file_offset(edit_unit)?;
        return Some(Landing {
            offset,
            edit_unit,
            split: in_clip(offset),
        });
    }
    let clip = clip_value?;
    let Essence::Pcm { sample_rate, .. } = tracks[clip.port].essence else {
        return None;
    };
    let block = sample_block(tracks[clip.port].essence)?;
    let sample =
        u64::try_from(u128::from(target_ns) * u128::from(sample_rate) / 1_000_000_000).ok()?;
    let offset = clip.at.checked_add(sample.checked_mul(block)?)?;
    Some(Landing {
        offset,
        edit_unit: sample,
        split: Some(in_clip(offset)?),
    })
}

#[derive(Debug)]
pub struct MxfDemuxN {
    buf: Vec<u8>,
    buf_at: u64,
    tracks: Vec<EssenceTrack>,
    port_caps: Vec<Caps>,
    bus: Option<BusHandle>,
    emitted: u64,
    seek: DemuxSeek<Unit>,
    tail: TailRead,
    index_probes: Vec<Range<u64>>,
    read: MxfRead,
    index: MxfIndex,
    reordering: Reordering,
    // stored edit units sent per picture port, samples per sound port
    positions: Vec<u64>,
    partition_at: Option<u64>,
    split: Option<SplitValue>,
    clip_value: Option<SplitValue>,
    own_seek_landing: Option<u64>,
    indexed_seek_landing: Option<Landing>,
    duration_ns: Option<u64>,
    // the one port's track, found once the header is read
    selection: Option<TrackSelection>,
}

impl MxfDemuxN {
    pub fn new(ports: Vec<MxfTrackInfo>) -> Self {
        assert!(
            !ports.is_empty(),
            "MxfDemuxN needs at least one output port"
        );
        let port_caps = ports.iter().map(|port| port.caps.clone()).collect();
        let tracks = ports.into_iter().map(|port| port.track).collect();
        Self::with_ports(port_caps, tracks, None)
    }

    fn selecting(stream: MxfStream) -> Self {
        let selection = TrackSelection {
            stream,
            track_number: None,
        };
        Self::with_ports(
            Vec::from([stream.first_placeholder_caps()]),
            Vec::new(),
            Some(selection),
        )
    }

    fn with_ports(
        port_caps: Vec<Caps>,
        tracks: Vec<EssenceTrack>,
        selection: Option<TrackSelection>,
    ) -> Self {
        Self {
            buf: Vec::new(),
            buf_at: 0,
            positions: alloc::vec![0; port_caps.len()],
            port_caps,
            tracks,
            bus: None,
            emitted: 0,
            seek: DemuxSeek::default(),
            tail: TailRead::default(),
            index_probes: Vec::new(),
            read: MxfRead::Head,
            index: MxfIndex::default(),
            reordering: Reordering::default(),
            partition_at: None,
            split: None,
            clip_value: None,
            own_seek_landing: None,
            indexed_seek_landing: None,
            duration_ns: None,
            selection,
        }
    }

    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn port_of(&self, track_number: u32) -> Option<usize> {
        self.tracks
            .iter()
            .position(|track| track.track_number == track_number)
    }

    fn picture_rate(&self) -> Option<Rational> {
        self.tracks
            .iter()
            .find(|track| is_video(track))
            .map(|track| track.edit_rate)
    }

    fn delay_ns(&self) -> u64 {
        self.picture_rate()
            .and_then(|rate| rate.units_to_ns(self.reordering.delay))
            .unwrap_or(0)
    }

    async fn announce(
        &mut self,
        header_tracks: &[EssenceTrack],
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        if let Some(selection) = self.selection {
            let track = header_tracks
                .iter()
                .find(|track| selection.picks(track))
                .ok_or_else(|| {
                    malformed(&format!(
                        "mxf: no {} track with track-number {}",
                        selection.stream.name(),
                        selection.track_number.unwrap_or(0)
                    ))
                })?;
            self.tracks = Vec::from([*track]);
        }
        self.duration_ns = self
            .tracks
            .iter()
            .filter_map(|track| track.edit_rate.units_to_ns(track.duration?))
            .max();
        self.seek.set_duration(self.duration_ns);
        if let Some(bus) = &self.bus {
            let streams = self
                .tracks
                .iter()
                .map(|track| {
                    let stream_type = match is_video(track) {
                        true => StreamType::Video,
                        false => StreamType::Audio,
                    };
                    Stream::new(
                        format!("mxf-track-{}", track.track_number),
                        stream_type,
                        track_caps(track),
                    )
                })
                .collect();
            bus.try_post(BusMessage::StreamCollection(StreamCollection::new(
                "mxf-0", streams,
            )));
            // the runner queries a multi-output demux for its duration, never a one-port element
            if let (Some(duration_ns), Some(_)) = (self.duration_ns, self.selection) {
                bus.try_post(BusMessage::DurationChanged { duration_ns });
            }
        }
        for (port, track) in self.tracks.iter().enumerate() {
            out.push_to(port, PipelinePacket::CapsChanged(track_caps(track)))
                .await?;
        }
        Ok(())
    }

    fn poll_seek(&mut self) {
        if matches!(
            self.read,
            MxfRead::Tail { .. } | MxfRead::Index { .. } | MxfRead::Rewind
        ) {
            return;
        }
        if !matches!(self.read, MxfRead::Klvs { .. }) {
            self.seek.poll_request();
            return;
        }
        let delay_ns = self.delay_ns();
        let stream_len = self.seek.upstream_len();
        let (tracks, index, clip_value) = (&self.tracks, &self.index, self.clip_value);
        let mut found = None;
        let taken = self.seek.poll_request_indexed(|target_ns| {
            found = landing(
                tracks,
                index,
                clip_value,
                target_ns.saturating_sub(delay_ns),
            )
            .filter(|landing| stream_len.is_none_or(|len| landing.offset < len));
            found.map(|landing| landing.offset)
        });
        if taken {
            self.indexed_seek_landing = found;
        }
    }

    async fn advance(
        &mut self,
        stream_ended: bool,
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        if self.read == MxfRead::Head {
            let Some(metadata) = header_partition_metadata(&self.buf) else {
                if self.buf.len() > MAX_HEADER_BYTES {
                    return Err(malformed("mxf header metadata too large"));
                }
                return Ok(());
            };
            self.announce(&metadata.tracks, out).await?;
            let seekable = self.seek.upstream_len().is_some() && self.seek.takes_seeks_after_end();
            self.read = match seekable {
                true => MxfRead::Tail { expanded: false },
                false => MxfRead::Whole,
            };
            if seekable {
                self.tail.takes_bytes(&self.seek, &[]);
                return Ok(());
            }
        }
        match self.read {
            MxfRead::Index { next, end } => {
                if self.probe_partition(next, end) {
                    self.next_index_probe();
                }
                Ok(())
            }
            MxfRead::Head | MxfRead::Tail { .. } | MxfRead::Rewind | MxfRead::Whole => Ok(()),
            MxfRead::Klvs { next } => self.emit_klvs(next, stream_ended, out).await,
        }
    }

    // Record the partitions already buffered with the tail and queue the ones before it.
    fn read_tail(&mut self, tail: &[u8], tail_at: u64, rip_len: u64) -> Option<()> {
        let start = tail.len().checked_sub(usize::try_from(rip_len).ok()?)?;
        let rip_at = tail_at.checked_add(start as u64)?;
        let (klv, value) = whole_klv(&tail[start..]).ok()?;
        if key_kind(&klv.key) != KeyKind::RandomIndexPack || klv.total_len()? != rip_len {
            return None;
        }
        let mut partitions = random_index_pack(value)?;
        partitions.retain(|at| *at < rip_at);
        partitions.sort_unstable();
        partitions.dedup();
        let before_tail = partitions.partition_point(|at| *at < tail_at);
        partitions.push(rip_at);
        for pair in partitions.windows(2).skip(before_tail) {
            let bytes = &tail[(pair[0] - tail_at) as usize..(pair[1] - tail_at) as usize];
            self.partition_at = None;
            if read_klv(bytes).is_ok_and(|klv| matches!(key_kind(&klv.key), KeyKind::Partition(_)))
            {
                self.observe(bytes, pair[0]);
            }
        }
        self.index_probes = partitions
            .windows(2)
            .take(before_tail)
            .map(|pair| pair[0]..pair[1])
            .rev()
            .collect();
        Some(())
    }

    fn next_index_probe(&mut self) {
        self.partition_at = None;
        let offset = if let Some(range) = self.index_probes.pop() {
            self.read = MxfRead::Index {
                next: range.start,
                end: range.end,
            };
            range.start
        } else {
            self.read = MxfRead::Rewind;
            0
        };
        self.own_seek_landing = Some(offset);
        self.seek.seek_upstream(offset);
    }

    // Probe until the first essence key or the end of the partition. Only
    // partition packs and index segments need their whole KLV values.
    fn probe_partition(&mut self, mut next: u64, end: u64) -> bool {
        while next < end && self.skip_to(next) {
            let klv = match read_klv(&self.buf) {
                Ok(klv) => klv,
                Err(KlvError::Truncated) => break,
                Err(KlvError::Malformed) => return true,
            };
            let Some(after) = klv.total_len().and_then(|len| next.checked_add(len)) else {
                return true;
            };
            if after > end {
                return true;
            }
            let kind = key_kind(&klv.key);
            if self.partition_at.is_none() && !matches!(kind, KeyKind::Partition(_)) {
                return true;
            }
            match kind {
                KeyKind::Essence { track_number } => {
                    self.place_essence(track_number, &klv, next);
                    return true;
                }
                KeyKind::Partition(_) if self.partition_at.is_some() => return true,
                KeyKind::Partition(_) | KeyKind::IndexSegment => {
                    if klv.value_len > MAX_PICTURE_BYTES {
                        return true;
                    }
                    if !self.note_buffered_klv(next) {
                        break;
                    }
                }
                KeyKind::EncryptedEssence | KeyKind::RandomIndexPack => return true,
                _ => {}
            }
            next = after;
        }
        self.read = MxfRead::Index { next, end };
        next == end
    }

    // records the partitions, index segments and essence starts of whole klvs in `bytes`, which begin at file byte `at`
    fn observe(&mut self, bytes: &[u8], mut at: u64) {
        let mut rest = bytes;
        while let Ok((klv, value)) = whole_klv(rest) {
            self.note_klv(&klv, value, at);
            let total = klv.header_len + value.len();
            rest = &rest[total..];
            let Some(next) = at.checked_add(total as u64) else {
                break;
            };
            at = next;
        }
    }

    fn note_buffered_klv(&mut self, at: u64) -> bool {
        let buf = core::mem::take(&mut self.buf);
        let complete = if let Ok((klv, value)) = whole_klv(&buf) {
            self.note_klv(&klv, value, at);
            true
        } else {
            false
        };
        self.buf = buf;
        complete
    }

    fn note_klv(&mut self, klv: &Klv, value: &[u8], at: u64) {
        match key_kind(&klv.key) {
            KeyKind::Partition(_) => {
                if let Some(pack) = partition_pack(value) {
                    self.index.add_partition(at, &pack);
                    self.partition_at = Some(at);
                }
            }
            KeyKind::IndexSegment => {
                if let Some(segment) = index_segment(value) {
                    if matches!(self.read, MxfRead::Klvs { .. }) {
                        self.reordering.extend(&segment);
                    }
                    self.index.add_segment(segment);
                }
            }
            KeyKind::Essence { track_number } => self.place_essence(track_number, klv, at),
            _ => {}
        }
    }

    fn place_essence(&mut self, track_number: u32, klv: &Klv, at: u64) {
        let Some(partition_at) = self.partition_at else {
            return;
        };
        let clip_wrapped = self
            .port_of(track_number)
            .is_some_and(|port| self.tracks[port].clip_wrapped);
        let essence_at = match clip_wrapped {
            true => at + klv.header_len as u64,
            false => at,
        };
        self.index.place_essence(partition_at, essence_at);
    }

    fn start_streaming(&mut self) {
        self.reordering = Reordering::of(&self.index);
        self.read = MxfRead::Klvs { next: 0 };
        self.buf.clear();
        self.buf_at = 0;
    }

    fn skip_to(&mut self, next: u64) -> bool {
        let passed = usize::try_from(next.saturating_sub(self.buf_at))
            .unwrap_or(usize::MAX)
            .min(self.buf.len());
        self.buf.drain(..passed);
        self.buf_at += passed as u64;
        self.buf_at >= next
    }

    async fn emit_klvs(
        &mut self,
        mut next: u64,
        stream_ended: bool,
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        // the seek is taken at the next packet, and the klvs after it come from where it lands
        while !(self.seek.app_seek_pending() && !self.seek.is_seeking()) {
            if !self.skip_to(next) {
                break;
            }
            if let Some(split) = self.split {
                let sent = match is_video(&self.tracks[split.port]) {
                    true => self.send_clip_picture(split, stream_ended, out).await?,
                    false => self.send_sound(split, stream_ended, out).await?,
                };
                if sent == 0 {
                    break;
                }
                next += sent;
                continue;
            }
            let klv = match read_klv(&self.buf) {
                Ok(klv) => klv,
                Err(KlvError::Truncated) => break,
                Err(KlvError::Malformed) => {
                    return Err(malformed("mxf: no klv key where one starts"))
                }
            };
            let after = klv
                .total_len()
                .and_then(|len| next.checked_add(len))
                .ok_or_else(|| malformed("mxf: klv length overflows"))?;
            match key_kind(&klv.key) {
                KeyKind::EncryptedEssence => return Err(malformed("mxf: encrypted essence")),
                KeyKind::Essence { track_number } => {
                    self.place_essence(track_number, &klv, next);
                    let Some(port) = self.port_of(track_number) else {
                        next = after;
                        continue;
                    };
                    let track = self.tracks[port];
                    if let Essence::Pcm {
                        format,
                        channels,
                        sample_rate,
                        packing: PcmPacking::D10Aes3,
                    } = track.essence
                    {
                        if klv.value_len > MAX_D10_SOUND_BYTES {
                            return Err(malformed("mxf: D-10 sound klv too large"));
                        }
                        let Ok((_, value)) = whole_klv(&self.buf) else {
                            if stream_ended {
                                return Err(malformed("mxf: truncated D-10 sound klv"));
                            }
                            break;
                        };
                        let (data, samples) = d10_pcm(value, format, channels)
                            .ok_or_else(|| malformed("mxf: malformed D-10 sound element"))?;
                        if samples != 0 {
                            self.send_pcm(port, data, samples, sample_rate, out).await?;
                        }
                        next = after;
                        continue;
                    }
                    if !is_video(&track) || track.clip_wrapped {
                        let at = next + klv.header_len as u64;
                        let split = SplitValue {
                            port,
                            at,
                            end: after,
                        };
                        if track.clip_wrapped && self.clip_value.is_none() {
                            self.clip_value = Some(split);
                        }
                        self.split = Some(split);
                        next = at;
                        continue;
                    }
                    if klv.value_len > MAX_PICTURE_BYTES {
                        return Err(malformed("mxf: picture klv too large"));
                    }
                    let Ok((_, value)) = whole_klv(&self.buf) else {
                        break;
                    };
                    let data = Vec::from(value);
                    self.send_picture(port, data, out).await?;
                    next = after;
                }
                KeyKind::Partition(_) | KeyKind::IndexSegment => {
                    if klv.value_len > MAX_PICTURE_BYTES {
                        return Err(malformed("mxf: index klv too large"));
                    }
                    if !self.note_buffered_klv(next) {
                        break;
                    }
                    next = after;
                }
                _ => next = after,
            }
        }
        self.read = MxfRead::Klvs { next };
        Ok(())
    }

    async fn send_picture(
        &mut self,
        port: usize,
        data: Vec<u8>,
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        let track = self.tracks[port];
        let Essence::Video { codec, .. } = track.essence else {
            unreachable!("pictures come from picture tracks");
        };
        let stored = self.positions[port];
        self.positions[port] += 1;
        let rate = track.edit_rate;
        let display = self.reordering.display(stored) + self.reordering.delay;
        // mpeg-2 open gop i-frames carry no random access flag, but decode on from their sequence header
        let keyframe = self
            .index
            .entry(stored)
            .is_some_and(|entry| entry.random_access)
            || crate::annexb::au_is_keyframe(codec, &data);
        let units_ns = |units: u64| rate.units_to_ns(units).unwrap_or(u64::MAX);
        let timing = FrameTiming {
            pts_ns: units_ns(display),
            dts_ns: units_ns(stored),
            duration_ns: units_ns(1),
            keyframe,
            ..FrameTiming::default()
        };
        self.send(Unit { port, data, timing }, keyframe, out).await
    }

    // the next edit unit of a clip-wrapped picture value, cut where the index puts the one after it
    async fn send_clip_picture(
        &mut self,
        split: SplitValue,
        stream_ended: bool,
        out: &mut dyn MultiOutputSink,
    ) -> Result<u64, G2gError> {
        let following = self.positions[split.port] + 1;
        let end = self
            .index
            .file_offset(following)
            .filter(|end| *end < split.end)
            .unwrap_or(split.end);
        let len = end
            .checked_sub(self.buf_at)
            .filter(|len| *len > 0 && *len <= MAX_PICTURE_BYTES)
            .ok_or_else(|| malformed("mxf: clip-wrapped picture outside its index"))?;
        if (self.buf.len() as u64) < len {
            if stream_ended {
                self.split = None;
            }
            return Ok(0);
        }
        let data = Vec::from(&self.buf[..len as usize]);
        if end == split.end {
            self.split = None;
        }
        self.send_picture(split.port, data, out).await?;
        Ok(len)
    }

    // whole sample blocks of the current sound value, at most one piece, returning the bytes passed
    async fn send_sound(
        &mut self,
        split: SplitValue,
        stream_ended: bool,
        out: &mut dyn MultiOutputSink,
    ) -> Result<u64, G2gError> {
        let essence = self.tracks[split.port].essence;
        let Essence::Pcm { sample_rate, .. } = essence else {
            unreachable!("sound values come from sound tracks");
        };
        let block = sample_block(essence).ok_or_else(|| malformed("mxf: no pcm sample size"))?;
        let left = split.end - self.buf_at;
        if left == 0 {
            self.split = None;
            return Ok(0);
        }
        let piece = (u64::from(sample_rate / SOUND_PIECES_PER_SECOND) * block)
            .max(block)
            .min(left);
        let available = (self.buf.len() as u64).min(left);
        let take = match (available >= piece, stream_ended) {
            (true, _) => piece,
            (false, true) => available,
            (false, false) => return Ok(0),
        };
        let take = match take == left {
            true => take,
            false => take - take % block,
        };
        if take == 0 {
            self.split = None;
            return Ok(0);
        }
        if take == left {
            self.split = None;
        }
        let data = Vec::from(&self.buf[..take as usize]);
        self.send_pcm(split.port, data, take / block, sample_rate, out)
            .await?;
        Ok(take)
    }

    async fn send_pcm(
        &mut self,
        port: usize,
        data: Vec<u8>,
        sample_count: u64,
        sample_rate: u32,
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        let samples = self.positions[port];
        self.positions[port] = samples
            .checked_add(sample_count)
            .ok_or_else(|| malformed("mxf: sound sample position overflows"))?;
        let pts_ns = self
            .delay_ns()
            .saturating_add(samples_ns(samples, sample_rate));
        let timing = FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns: samples_ns(sample_count, sample_rate),
            keyframe: true,
            ..FrameTiming::default()
        };
        self.send(Unit { port, data, timing }, true, out).await
    }

    async fn send(
        &mut self,
        unit: Unit,
        resync_point: bool,
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        let pts_ns = unit.timing.pts_ns;
        let port = unit.port;
        self.seek.note_track_unit(port as u64, pts_ns);
        let admitted = self.seek.admit(port, pts_ns, resync_point, unit);
        if let Some(seek_segment) = admitted.segment {
            for port in 0..self.port_caps.len() {
                out.push_to(port, PipelinePacket::Segment(seek_segment))
                    .await?;
            }
        }
        for unit in admitted {
            let frame = Frame::from_system_bytes(unit.data, unit.timing, self.emitted);
            self.emitted += 1;
            out.push_to(unit.port, PipelinePacket::DataFrame(frame))
                .await?;
        }
        Ok(())
    }

    // the read after a seek's flush: on from an index landing, or the file again from its start
    fn restart_read(&mut self, landing: Option<Landing>) {
        self.buf.clear();
        self.split = None;
        if !matches!(self.read, MxfRead::Klvs { .. }) {
            self.buf_at = 0;
            return;
        }
        let Some(landing) = landing else {
            self.buf_at = 0;
            self.read = MxfRead::Klvs { next: 0 };
            self.positions.iter_mut().for_each(|position| *position = 0);
            return;
        };
        self.buf_at = landing.offset;
        self.read = MxfRead::Klvs {
            next: landing.offset,
        };
        self.split = landing.split;
        self.partition_at = self
            .index
            .partitions
            .iter()
            .filter(|place| place.at <= landing.offset)
            .map(|place| place.at)
            .max();
        let picture_rate = self.picture_rate();
        for (port, track) in self.tracks.iter().enumerate() {
            self.positions[port] = match (track.essence, picture_rate) {
                (
                    Essence::Pcm {
                        sample_rate,
                        packing,
                        ..
                    },
                    Some(rate),
                ) => {
                    let denominator = u128::from(rate.numerator);
                    let mut numerator = u128::from(landing.edit_unit)
                        * u128::from(sample_rate)
                        * u128::from(rate.denominator);
                    // ST 331's five-frame cadence rounds each edit boundary to the nearest sample.
                    if packing == PcmPacking::D10Aes3 {
                        numerator += denominator / 2;
                    }
                    u64::try_from(numerator / denominator).unwrap_or(u64::MAX)
                }
                _ => landing.edit_unit,
            };
        }
    }

    fn end_tail_read(&mut self, expanded: bool) {
        let Some(tail) = self.tail.finish_read() else {
            return;
        };
        if let (Some(stream_len), Some(rip_len)) =
            (self.seek.upstream_len(), random_index_pack_len(&tail))
        {
            if let Some(rip_at) = stream_len.checked_sub(rip_len) {
                if rip_len <= tail.len() as u64 {
                    let tail_at = stream_len.saturating_sub(tail.len() as u64);
                    let _ = self.read_tail(&tail, tail_at, rip_len);
                } else if !expanded && rip_len <= MAX_HEADER_BYTES as u64 {
                    self.read = MxfRead::Tail { expanded: true };
                    self.tail.seek_from(&self.seek, rip_at);
                    return;
                }
            }
        }
        self.next_index_probe();
    }

    // the file held whole when its source cannot seek, read once for its index then sent
    async fn emit_whole(&mut self, out: &mut dyn MultiOutputSink) -> Result<(), G2gError> {
        if self.read == MxfRead::Head {
            let metadata = header_partition_metadata(&self.buf)
                .ok_or_else(|| malformed("mxf: no header partition metadata"))?;
            self.announce(&metadata.tracks, out).await?;
        }
        let whole = core::mem::take(&mut self.buf);
        self.observe(&whole, 0);
        self.start_streaming();
        self.buf = whole;
        self.emit_klvs(0, true, out).await
    }
}

impl MultiOutputElement for MxfDemuxN {
    fn set_bus(&mut self, bus: BusHandle) {
        self.bus = Some(bus);
    }

    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&input_caps())
    }

    fn caps_constraint_as_input(&self) -> CapsConstraint<'_> {
        CapsConstraint::Produces(CapsSet::one(input_caps()))
    }

    fn port_output_caps(&self, port: usize) -> Option<Caps> {
        self.port_caps.get(port).cloned()
    }

    fn port_caps_constraint(&self, port: usize) -> Option<CapsConstraint<'_>> {
        let Some(selection) = self.selection else {
            return self
                .port_output_caps(port)
                .map(|caps| CapsConstraint::Produces(CapsSet::one(caps)));
        };
        (port == 0).then(|| {
            CapsConstraint::Produces(CapsSet::from_alternatives(
                selection.stream.placeholder_caps(),
            ))
        })
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "MXF demuxer",
            "Codec/Demuxer",
            "Demuxes an MXF file into the essence tracks its ports select",
            crate::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        match self.selection {
            Some(_) => ONE_PORT_PROPS,
            None => &[],
        }
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let Some(selection) = &mut self.selection else {
            return Err(PropError::Unknown);
        };
        match name {
            "stream" => {
                let name = value.as_str().ok_or(PropError::Type)?;
                selection.stream = MxfStream::from_name(name).ok_or(PropError::Value)?;
                self.port_caps[0] = selection.stream.first_placeholder_caps();
                Ok(())
            }
            "track-number" => {
                let number = value.as_uint().ok_or(PropError::Type)?;
                let number = u32::try_from(number).map_err(|_| PropError::Value)?;
                selection.track_number = (number != 0).then_some(number);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let selection = self.selection?;
        match name {
            "stream" => Some(PropValue::Str(String::from(selection.stream.name()))),
            "track-number" => Some(PropValue::Uint(u64::from(
                selection.track_number.unwrap_or(0),
            ))),
            _ => None,
        }
    }

    fn seek_route(&self) -> Option<ElementSeekRoute> {
        Some(ElementSeekRoute::UpstreamBytes)
    }

    fn set_seek(&mut self, app: SeekController, upstream: SeekController) {
        self.seek.with(app, upstream);
    }

    fn query_duration(&self) -> Option<u64> {
        self.duration_ns
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if *absolute_caps != input_caps() {
            return Err(G2gError::CapsMismatch);
        }
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            self.seek.hold_at_segment_end().await;
            self.poll_seek();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    if matches!(self.read, MxfRead::Tail { .. })
                        && self.tail.takes_bytes(&self.seek, slice)
                    {
                        return Ok(());
                    }
                    if self.seek.dropping_input() || self.own_seek_landing.is_some() {
                        return Ok(());
                    }
                    self.buf.extend_from_slice(slice);
                    self.advance(false, out).await?;
                }
                PipelinePacket::Eos => {
                    match self.read {
                        MxfRead::Tail { expanded } => {
                            self.end_tail_read(expanded);
                            return Ok(());
                        }
                        MxfRead::Index { .. } => {
                            if self.own_seek_landing.is_none() {
                                self.next_index_probe();
                            }
                            return Ok(());
                        }
                        MxfRead::Rewind => return Ok(()),
                        _ => {}
                    }
                    let awaiting_flush =
                        self.seek.dropping_input() || self.own_seek_landing.is_some();
                    if self.seek.eos_before_flush(awaiting_flush).await? {
                        return Ok(());
                    }
                    if !self.seek.dropping_input() {
                        self.advance(true, out).await?;
                        if matches!(self.read, MxfRead::Head | MxfRead::Whole) {
                            self.emit_whole(out).await?;
                        }
                    }
                    self.seek
                        .end_ports_and_wait_for_seek(self.port_caps.len(), out)
                        .await?;
                    self.poll_seek();
                }
                PipelinePacket::Flush => {
                    if self.tail.takes_flush() {
                        return Ok(());
                    }
                    if let Some(landing) = self.own_seek_landing.take() {
                        if self.read == MxfRead::Rewind {
                            self.start_streaming();
                        } else {
                            self.buf.clear();
                            self.buf_at = landing;
                        }
                        return Ok(());
                    }
                    let landing = self
                        .indexed_seek_landing
                        .take()
                        .filter(|_| self.seek.dropping_input() && self.seek.keeps_state());
                    let forward = self.seek.on_flush();
                    self.restart_read(landing);
                    if forward {
                        for port in 0..self.port_caps.len() {
                            out.push_to(port, PipelinePacket::Flush).await?;
                        }
                    }
                }
                PipelinePacket::Segment(segment) => {
                    for port in 0..self.port_caps.len() {
                        out.push_to(port, PipelinePacket::Segment(segment)).await?;
                    }
                }
                _ => {}
            }
            Ok(())
        })
    }
}

static ONE_PORT_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "stream",
        PropKind::Str,
        "essence to emit: h264 | mpeg2 | jpeg2000 | dnxhd | prores-proxy | prores-lt | prores-standard | prores-hq | prores-4444 | prores-4444xq | S16LE | S24LE | S32LE",
    )
    .with_default("h264"),
    PropertySpec::new(
        "track-number",
        PropKind::Uint,
        "MXF track number of the track to emit among those carrying `stream` (0 = the first of them)",
    )
    .with_default("0"),
];

pub type MxfDemux = OnePortElement<MxfDemuxN>;

impl Default for MxfDemux {
    fn default() -> Self {
        Self::new()
    }
}

impl MxfDemux {
    pub fn new() -> Self {
        Self::wrap(MxfDemuxN::selecting(DEFAULT_STREAM), "MxfDemux")
    }
}

impl PadTemplates for MxfDemux {
    fn pad_templates() -> Vec<PadTemplate> {
        let source = CapsSet::from_alternatives(
            MxfStream::all()
                .flat_map(MxfStream::placeholder_caps)
                .collect(),
        );
        Vec::from([
            PadTemplate::sink(CapsSet::one(input_caps())),
            PadTemplate::source(source),
        ])
    }
}
