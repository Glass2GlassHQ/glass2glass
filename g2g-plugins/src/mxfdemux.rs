// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::vec::Vec;

use g2g_core::caps::framerate_q16_from_fraction;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::log::Target;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{ElementSeekRoute, SeekController};
use g2g_core::{
    BusHandle, BusMessage, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    Dim, G2gError, MultiOutputElement, MultiOutputSink, Rate, Stream, StreamCollection, StreamType,
};

use crate::demuxseek::{DemuxSeek, TailRead};
use crate::mxf::{
    header_partition_metadata, index_segment, key_kind, partition_pack, random_index_pack,
    random_index_pack_len, read_klv, whole_klv, Essence, EssenceTrack, IndexEntry, IndexSegment,
    KeyKind, Klv, KlvError, PartitionPack, Rational,
};

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

    // the file byte holding `stream_offset` of the essence container, as ffmpeg's mxf_absolute_bodysid_offset finds it
    fn file_offset(&self, stream_offset: u64) -> Option<u64> {
        let place = self
            .partitions
            .iter()
            .filter(|place| place.body_sid != 0 && place.body_offset <= stream_offset)
            .max_by_key(|place| place.body_offset)?;
        place
            .essence_at?
            .checked_add(stream_offset - place.body_offset)
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

    fn stream_offset(&self, unit: u64) -> Option<u64> {
        let segment = self.segment_of(unit)?;
        if segment.edit_unit_byte_count > 0 {
            return u64::from(segment.edit_unit_byte_count).checked_mul(unit);
        }
        Some(self.entry(unit)?.stream_offset)
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
            let display = segment.start + at as u64;
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
    // the tail is read through a byte seek for the footer's index, then the file again from byte 0
    Tail,
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
        let offset = index.file_offset(index.stream_offset(edit_unit)?)?;
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
}

impl MxfDemuxN {
    pub fn new(ports: Vec<MxfTrackInfo>) -> Self {
        assert!(
            !ports.is_empty(),
            "MxfDemuxN needs at least one output port"
        );
        Self {
            buf: Vec::new(),
            buf_at: 0,
            port_caps: ports.iter().map(|port| port.caps.clone()).collect(),
            positions: alloc::vec![0; ports.len()],
            tracks: ports.into_iter().map(|port| port.track).collect(),
            bus: None,
            emitted: 0,
            seek: DemuxSeek::default(),
            tail: TailRead::default(),
            read: MxfRead::Head,
            index: MxfIndex::default(),
            reordering: Reordering::default(),
            partition_at: None,
            split: None,
            clip_value: None,
            own_seek_landing: None,
            indexed_seek_landing: None,
            duration_ns: None,
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

    async fn announce(&mut self, out: &mut dyn MultiOutputSink) -> Result<(), G2gError> {
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
        }
        self.duration_ns = self
            .tracks
            .iter()
            .filter_map(|track| track.edit_rate.units_to_ns(track.duration?))
            .max();
        self.seek.set_duration(self.duration_ns);
        for port in 0..self.tracks.len() {
            let caps = track_caps(&self.tracks[port]);
            out.push_to(port, PipelinePacket::CapsChanged(caps)).await?;
        }
        Ok(())
    }

    fn poll_seek(&mut self) {
        if self.own_seek_landing.is_some() || self.tail.in_flight() {
            return;
        }
        if !matches!(self.read, MxfRead::Klvs { .. }) {
            self.seek.poll_request();
            return;
        }
        let delay_ns = self.delay_ns();
        let (tracks, index, clip_value) = (&self.tracks, &self.index, self.clip_value);
        let mut found = None;
        let taken = self.seek.poll_request_indexed(|target_ns| {
            found = landing(
                tracks,
                index,
                clip_value,
                target_ns.saturating_sub(delay_ns),
            );
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
            if header_partition_metadata(&self.buf).is_none() {
                if self.buf.len() > MAX_HEADER_BYTES {
                    return Err(malformed("mxf header metadata too large"));
                }
                return Ok(());
            }
            self.announce(out).await?;
            let seekable = self.seek.upstream_len().is_some() && self.seek.takes_seeks_after_end();
            self.read = match seekable {
                true => MxfRead::Tail,
                false => MxfRead::Whole,
            };
            if self.read == MxfRead::Tail {
                self.tail.takes_bytes(&self.seek, &[]);
                return Ok(());
            }
        }
        match self.read {
            MxfRead::Head | MxfRead::Tail | MxfRead::Whole => Ok(()),
            MxfRead::Klvs { next } => self.emit_klvs(next, stream_ended, out).await,
        }
    }

    // the partitions and index segments of the file's last bytes, which start at `tail_at`
    fn read_tail(&mut self, tail: &[u8], tail_at: u64) {
        let partitions = random_index_pack_len(tail)
            .and_then(|len| usize::try_from(len).ok())
            .and_then(|len| tail.len().checked_sub(len))
            .and_then(|rip_at| whole_klv(&tail[rip_at..]).ok())
            .filter(|(klv, _)| key_kind(&klv.key) == KeyKind::RandomIndexPack)
            .and_then(|(_, value)| random_index_pack(value))
            .unwrap_or_default();
        let first_in_tail = partitions.iter().filter(|at| **at >= tail_at).min();
        if let Some(first) = first_in_tail {
            let start = (first - tail_at) as usize;
            self.observe(&tail[start..], *first);
        }
    }

    // records the partitions, index segments and essence starts of whole klvs in `bytes`, which begin at file byte `at`
    fn observe(&mut self, bytes: &[u8], mut at: u64) {
        let mut rest = bytes;
        while let Ok((klv, value)) = whole_klv(rest) {
            self.note_klv(&klv, value, at);
            let total = klv.header_len + value.len();
            rest = &rest[total..];
            at += total as u64;
        }
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
                    self.reordering.extend(&segment);
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

    async fn emit_klvs(
        &mut self,
        mut next: u64,
        stream_ended: bool,
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        // the seek is taken at the next packet, and the klvs after it come from where it lands
        while !(self.seek.app_seek_pending() && !self.seek.is_seeking()) {
            let passed = usize::try_from(next.saturating_sub(self.buf_at))
                .unwrap_or(usize::MAX)
                .min(self.buf.len());
            self.buf.drain(..passed);
            self.buf_at += passed as u64;
            if next > self.buf_at {
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
            let total = klv
                .total_len()
                .ok_or_else(|| malformed("mxf: klv length overflows"))?;
            match key_kind(&klv.key) {
                KeyKind::EncryptedEssence => return Err(malformed("mxf: encrypted essence")),
                KeyKind::Essence { track_number } => {
                    self.place_essence(track_number, &klv, next);
                    let Some(port) = self.port_of(track_number) else {
                        next += total;
                        continue;
                    };
                    let track = self.tracks[port];
                    if !is_video(&track) || track.clip_wrapped {
                        let at = next + klv.header_len as u64;
                        let split = SplitValue {
                            port,
                            at,
                            end: at + klv.value_len,
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
                    next += total;
                }
                KeyKind::Partition(_) | KeyKind::IndexSegment => {
                    if klv.value_len > MAX_PICTURE_BYTES {
                        return Err(malformed("mxf: index klv too large"));
                    }
                    let Ok((klv, value)) = whole_klv(&self.buf) else {
                        break;
                    };
                    let value = Vec::from(value);
                    self.note_klv(&klv, &value, next);
                    next += total;
                }
                _ => next += total,
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
            .stream_offset(following)
            .and_then(|offset| self.index.file_offset(offset))
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
        let samples = self.positions[split.port];
        let sample_count = take / block;
        self.positions[split.port] += sample_count;
        let pts_ns = self.delay_ns() + samples_ns(samples, sample_rate);
        let timing = FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns: samples_ns(sample_count, sample_rate),
            keyframe: true,
            ..FrameTiming::default()
        };
        let unit = Unit {
            port: split.port,
            data,
            timing,
        };
        self.send(unit, true, out).await?;
        Ok(take)
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
            for port in 0..self.tracks.len() {
                out.push_to(port, PipelinePacket::Segment(seek_segment))
                    .await?;
            }
        }
        for unit in admitted {
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(unit.data.into_boxed_slice())),
                timing: unit.timing,
                sequence: self.emitted,
                meta: Default::default(),
            };
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
                (Essence::Pcm { sample_rate, .. }, Some(rate)) => u64::try_from(
                    u128::from(landing.edit_unit)
                        * u128::from(sample_rate)
                        * u128::from(rate.denominator)
                        / u128::from(rate.numerator),
                )
                .unwrap_or(u64::MAX),
                _ => landing.edit_unit,
            };
        }
    }

    fn end_tail_read(&mut self) {
        let stream_len = self.seek.upstream_len();
        let Some(tail) = self.tail.take_tail(&self.seek) else {
            return;
        };
        let tail_at = stream_len.map_or(0, |len| len.saturating_sub(tail.len() as u64));
        self.read_tail(&tail, tail_at);
    }

    // the file held whole when its source cannot seek, read once for its index then sent
    async fn emit_whole(&mut self, out: &mut dyn MultiOutputSink) -> Result<(), G2gError> {
        if self.read == MxfRead::Head {
            if header_partition_metadata(&self.buf).is_none() {
                return Err(malformed("mxf: no header partition metadata"));
            }
            self.announce(out).await?;
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
                    if self.read == MxfRead::Tail && self.tail.takes_bytes(&self.seek, slice) {
                        return Ok(());
                    }
                    if self.seek.dropping_input() || self.own_seek_landing.is_some() {
                        return Ok(());
                    }
                    self.buf.extend_from_slice(slice);
                    self.advance(false, out).await?;
                }
                PipelinePacket::Eos => {
                    if self.read == MxfRead::Tail {
                        self.end_tail_read();
                        return Ok(());
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
                        .end_ports_and_wait_for_seek(self.tracks.len(), out)
                        .await?;
                    self.poll_seek();
                }
                PipelinePacket::Flush => {
                    if self.tail.takes_flush() {
                        if !self.tail.in_flight() {
                            self.start_streaming();
                        }
                        return Ok(());
                    }
                    if let Some(landing) = self.own_seek_landing.take() {
                        self.buf.clear();
                        self.buf_at = landing;
                        return Ok(());
                    }
                    let landing = self
                        .indexed_seek_landing
                        .take()
                        .filter(|_| self.seek.dropping_input() && self.seek.keeps_state());
                    let forward = self.seek.on_flush();
                    self.restart_read(landing);
                    if forward {
                        for port in 0..self.tracks.len() {
                            out.push_to(port, PipelinePacket::Flush).await?;
                        }
                    }
                }
                PipelinePacket::Segment(segment) => {
                    for port in 0..self.tracks.len() {
                        out.push_to(port, PipelinePacket::Segment(segment)).await?;
                    }
                }
                _ => {}
            }
            Ok(())
        })
    }
}
