// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec::Vec;

use g2g_core::{AudioFormat, ProResVariant, VideoCodec};

use crate::bereader::{be_u16, be_u32, be_u64};

pub(crate) type Label = [u8; 16];

// smpte labels differ in their registry version byte between writers
const LABEL_VERSION_BYTE: usize = 7;

fn label_starts_with(label: &[u8], prefix: &[u8]) -> bool {
    label.len() >= prefix.len()
        && label
            .iter()
            .zip(prefix)
            .enumerate()
            .all(|(at, (a, b))| at == LABEL_VERSION_BYTE || a == b)
}

const PARTITION_PACK: [u8; 13] = [
    0x06, 0x0e, 0x2b, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0d, 0x01, 0x02, 0x01, 0x01,
];
const PRIMER_PACK_KIND: u8 = 0x05;
const INDEX_SEGMENT_KIND: u8 = 0x10;
const RANDOM_INDEX_PACK_KIND: u8 = 0x11;
const INDEX_SEGMENT: [u8; 13] = [
    0x06, 0x0e, 0x2b, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0d, 0x01, 0x02, 0x01, 0x01,
];
const HEADER_SET: [u8; 14] = [
    0x06, 0x0e, 0x2b, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0d, 0x01, 0x01, 0x01, 0x01, 0x01,
];
const FILL: [u8; 16] = [
    0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x01, 0x03, 0x01, 0x02, 0x10, 0x01, 0x00, 0x00, 0x00,
];
const ESSENCE_ELEMENT: [u8; 12] = [
    0x06, 0x0e, 0x2b, 0x34, 0x01, 0x02, 0x01, 0x01, 0x0d, 0x01, 0x03, 0x01,
];
const ENCRYPTED_TRIPLET: [u8; 16] = [
    0x06, 0x0e, 0x2b, 0x34, 0x02, 0x04, 0x01, 0x07, 0x0d, 0x01, 0x03, 0x01, 0x02, 0x7e, 0x01, 0x00,
];
// the first byte of every key, smpte's object identifier
const KEY_START: u8 = 0x06;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PartitionKind {
    Header,
    Body,
    Footer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeyKind {
    Partition(PartitionKind),
    Primer,
    HeaderSet(u8),
    IndexSegment,
    RandomIndexPack,
    Fill,
    Essence { track_number: u32 },
    EncryptedEssence,
    Other,
}

pub(crate) fn key_kind(key: &Label) -> KeyKind {
    if label_starts_with(key, &PARTITION_PACK) {
        return match key[13] {
            0x02 => KeyKind::Partition(PartitionKind::Header),
            0x03 => KeyKind::Partition(PartitionKind::Body),
            0x04 => KeyKind::Partition(PartitionKind::Footer),
            PRIMER_PACK_KIND => KeyKind::Primer,
            RANDOM_INDEX_PACK_KIND => KeyKind::RandomIndexPack,
            _ => KeyKind::Other,
        };
    }
    if label_starts_with(key, &INDEX_SEGMENT) && key[13] == INDEX_SEGMENT_KIND {
        return KeyKind::IndexSegment;
    }
    if label_starts_with(key, &HEADER_SET) {
        return KeyKind::HeaderSet(key[14]);
    }
    if label_starts_with(key, &FILL) {
        return KeyKind::Fill;
    }
    if label_starts_with(key, &ENCRYPTED_TRIPLET) {
        return KeyKind::EncryptedEssence;
    }
    if label_starts_with(key, &ESSENCE_ELEMENT) {
        let track_number = u32::from_be_bytes([key[12], key[13], key[14], key[15]]);
        return KeyKind::Essence { track_number };
    }
    KeyKind::Other
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Klv {
    pub key: Label,
    pub header_len: usize,
    pub value_len: u64,
}

impl Klv {
    pub(crate) fn total_len(&self) -> Option<u64> {
        self.value_len.checked_add(self.header_len as u64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KlvError {
    Truncated,
    Malformed,
}

const KEY_LEN: usize = 16;
// ber lengths past eight bytes overflow a u64
const MAX_BER_LENGTH_BYTES: usize = 8;

pub(crate) fn read_klv(data: &[u8]) -> Result<Klv, KlvError> {
    let key: Label = data
        .get(..KEY_LEN)
        .ok_or(KlvError::Truncated)?
        .try_into()
        .map_err(|_| KlvError::Truncated)?;
    if key[0] != KEY_START {
        return Err(KlvError::Malformed);
    }
    let first = *data.get(KEY_LEN).ok_or(KlvError::Truncated)?;
    if first < 0x80 {
        return Ok(Klv {
            key,
            header_len: KEY_LEN + 1,
            value_len: u64::from(first),
        });
    }
    let length_bytes = usize::from(first & 0x7f);
    if length_bytes == 0 || length_bytes > MAX_BER_LENGTH_BYTES {
        return Err(KlvError::Malformed);
    }
    let start = KEY_LEN + 1;
    let bytes = data
        .get(start..start + length_bytes)
        .ok_or(KlvError::Truncated)?;
    let value_len = bytes
        .iter()
        .fold(0u64, |length, byte| (length << 8) | u64::from(*byte));
    Ok(Klv {
        key,
        header_len: start + length_bytes,
        value_len,
    })
}

// a klv whose value lies wholly inside `data`, with that value
pub(crate) fn whole_klv(data: &[u8]) -> Result<(Klv, &[u8]), KlvError> {
    let klv = read_klv(data)?;
    let end = klv
        .total_len()
        .and_then(|end| usize::try_from(end).ok())
        .ok_or(KlvError::Malformed)?;
    let value = data.get(klv.header_len..end).ok_or(KlvError::Truncated)?;
    Ok((klv, value))
}

pub(crate) fn klvs(mut data: &[u8]) -> impl Iterator<Item = (Klv, &[u8])> {
    core::iter::from_fn(move || {
        let (klv, value) = whole_klv(data).ok()?;
        data = &data[klv.header_len + value.len()..];
        Some((klv, value))
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PartitionPack {
    pub op_atom: bool,
    pub header_byte_count: u64,
    pub body_offset: u64,
    pub body_sid: u32,
}

const OPERATIONAL_PATTERN_AT: usize = 64;
const OP_ATOM: [u8; 13] = [
    0x06, 0x0e, 0x2b, 0x34, 0x04, 0x01, 0x01, 0x02, 0x0d, 0x01, 0x02, 0x01, 0x10,
];

pub(crate) fn partition_pack(value: &[u8]) -> Option<PartitionPack> {
    let operational_pattern = value.get(OPERATIONAL_PATTERN_AT..OPERATIONAL_PATTERN_AT + 16)?;
    Some(PartitionPack {
        op_atom: label_starts_with(operational_pattern, &OP_ATOM),
        header_byte_count: be_u64(value, 32)?,
        body_offset: be_u64(value, 52)?,
        body_sid: be_u32(value, 60)?,
    })
}

const RANDOM_INDEX_ENTRY_LEN: usize = 12;
const RANDOM_INDEX_LENGTH_FIELD: usize = 4;

// the byte offset of every partition, in file order
pub(crate) fn random_index_pack(value: &[u8]) -> Option<Vec<u64>> {
    let entries = value.len().checked_sub(RANDOM_INDEX_LENGTH_FIELD)?;
    if entries % RANDOM_INDEX_ENTRY_LEN != 0 {
        return None;
    }
    value[..entries]
        .as_chunks::<RANDOM_INDEX_ENTRY_LEN>()
        .0
        .iter()
        .map(|entry| be_u64(entry, 4))
        .collect()
}

// the overall length the random index pack ends with, which locates it from the end of the file
pub(crate) fn random_index_pack_len(file_tail: &[u8]) -> Option<u64> {
    let at = file_tail.len().checked_sub(RANDOM_INDEX_LENGTH_FIELD)?;
    be_u32(file_tail, at).map(u64::from)
}

// ---- header metadata ----

type InstanceUid = [u8; 16];

const INSTANCE_UID_TAG: u16 = 0x3c0a;

struct LocalSet<'a> {
    kind: u8,
    items: Vec<(u16, &'a [u8])>,
}

impl<'a> LocalSet<'a> {
    fn uid(&self) -> Option<InstanceUid> {
        self.reference(INSTANCE_UID_TAG)
    }

    fn item(&self, tag: u16) -> Option<&'a [u8]> {
        self.items
            .iter()
            .find(|(item_tag, _)| *item_tag == tag)
            .map(|(_, value)| *value)
    }

    fn u8(&self, tag: u16) -> Option<u8> {
        self.item(tag)?.first().copied()
    }

    fn u32(&self, tag: u16) -> Option<u32> {
        be_u32(self.item(tag)?, 0)
    }

    fn i64(&self, tag: u16) -> Option<i64> {
        be_u64(self.item(tag)?, 0).map(|value| value as i64)
    }

    fn label(&self, tag: u16) -> Option<Label> {
        self.item(tag)?.get(..16)?.try_into().ok()
    }

    fn rational(&self, tag: u16) -> Option<Rational> {
        let value = self.item(tag)?;
        Rational::new(be_u32(value, 0)?, be_u32(value, 4)?)
    }

    fn references(&self, tag: u16) -> Vec<InstanceUid> {
        self.item(tag).map(batch).unwrap_or_default()
    }

    fn reference(&self, tag: u16) -> Option<InstanceUid> {
        self.item(tag)?.get(..16)?.try_into().ok()
    }
}

const BATCH_HEADER_LEN: usize = 8;

fn batch(value: &[u8]) -> Vec<InstanceUid> {
    let (Some(count), Some(item_len)) = (be_u32(value, 0), be_u32(value, 4)) else {
        return Vec::new();
    };
    if item_len != 16 {
        return Vec::new();
    }
    value
        .get(BATCH_HEADER_LEN..)
        .unwrap_or_default()
        .as_chunks::<16>()
        .0
        .iter()
        .take(count as usize)
        .copied()
        .collect()
}

fn local_set(kind: u8, value: &[u8]) -> Option<LocalSet<'_>> {
    let mut items = Vec::new();
    let mut at = 0;
    while at < value.len() {
        let tag = be_u16(value, at)?;
        let len = usize::from(be_u16(value, at + 2)?);
        let item = value.get(at + 4..at + 4 + len)?;
        items.push((tag, item));
        at += 4 + len;
    }
    Some(LocalSet { kind, items })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Rational {
    pub numerator: u32,
    pub denominator: u32,
}

impl Rational {
    fn new(numerator: u32, denominator: u32) -> Option<Self> {
        // mxf rationals are signed, a negative or zero part is no rate
        let positive = |part: u32| part > 0 && part <= i32::MAX as u32;
        (positive(numerator) && positive(denominator)).then_some(Rational {
            numerator,
            denominator,
        })
    }

    // nanoseconds from the start to edit unit `units`, None past u64
    pub(crate) fn units_to_ns(self, units: u64) -> Option<u64> {
        let ns = u128::from(units) * u128::from(self.denominator) * 1_000_000_000
            / u128::from(self.numerator);
        u64::try_from(ns).ok()
    }

    // the edit unit that covers `ns`
    pub(crate) fn ns_to_units(self, ns: u64) -> u64 {
        let units = u128::from(ns) * u128::from(self.numerator)
            / (u128::from(self.denominator) * 1_000_000_000);
        u64::try_from(units).unwrap_or(u64::MAX)
    }
}

const SOURCE_PACKAGE: u8 = 0x37;
const TIMELINE_TRACK: u8 = 0x3b;
const SEQUENCE: u8 = 0x0f;
const ESSENCE_CONTAINER_DATA: u8 = 0x23;
const MULTIPLE_DESCRIPTOR: u8 = 0x44;
const PICTURE_DESCRIPTORS: [u8; 4] = [0x27, 0x28, 0x29, 0x51];
const SOUND_DESCRIPTORS: [u8; 3] = [0x42, 0x47, 0x48];

const PACKAGE_UID_TAG: u16 = 0x4401;
const PACKAGE_TRACKS_TAG: u16 = 0x4403;
const PACKAGE_DESCRIPTOR_TAG: u16 = 0x4701;
const LINKED_PACKAGE_UID_TAG: u16 = 0x2701;
const TRACK_ID_TAG: u16 = 0x4801;
const TRACK_NUMBER_TAG: u16 = 0x4804;
const TRACK_SEQUENCE_TAG: u16 = 0x4803;
const EDIT_RATE_TAG: u16 = 0x4b01;
const COMPONENT_DURATION_TAG: u16 = 0x0202;
const SUB_DESCRIPTORS_TAG: u16 = 0x3f01;
const LINKED_TRACK_ID_TAG: u16 = 0x3006;
const CONTAINER_DURATION_TAG: u16 = 0x3002;
const ESSENCE_CONTAINER_TAG: u16 = 0x3004;
const STORED_WIDTH_TAG: u16 = 0x3203;
const STORED_HEIGHT_TAG: u16 = 0x3202;
const DISPLAY_WIDTH_TAG: u16 = 0x3209;
const DISPLAY_HEIGHT_TAG: u16 = 0x3208;
const FRAME_LAYOUT_TAG: u16 = 0x320c;
const PICTURE_CODING_TAG: u16 = 0x3201;
const AUDIO_SAMPLING_RATE_TAG: u16 = 0x3d03;
const CHANNEL_COUNT_TAG: u16 = 0x3d07;
const QUANTIZATION_BITS_TAG: u16 = 0x3d01;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Essence {
    Video {
        codec: VideoCodec,
        width: u32,
        height: u32,
    },
    Pcm {
        format: AudioFormat,
        channels: u8,
        sample_rate: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EssenceTrack {
    pub track_number: u32,
    pub edit_rate: Rational,
    pub duration: Option<u64>,
    pub essence: Essence,
    // one klv holds the whole track, which index offsets cut into edit units
    pub clip_wrapped: bool,
}

#[derive(Debug, Default)]
pub(crate) struct HeaderMetadata {
    pub tracks: Vec<EssenceTrack>,
    // tracks with an essence the demuxer has no caps for, by their essence container label
    pub unsupported: Vec<Label>,
}

pub(crate) fn header_metadata(data: &[u8], op_atom: bool) -> HeaderMetadata {
    let sets: Vec<LocalSet> = klvs(data)
        .filter_map(|(klv, value)| match key_kind(&klv.key) {
            KeyKind::HeaderSet(kind) => local_set(kind, value),
            _ => None,
        })
        .collect();
    let by_uid = |uid: &InstanceUid| sets.iter().find(|set| set.uid() == Some(*uid));
    let mut metadata = HeaderMetadata::default();
    for package in file_packages(&sets) {
        let descriptor = package
            .reference(PACKAGE_DESCRIPTOR_TAG)
            .and_then(|uid| by_uid(&uid));
        let Some(descriptor) = descriptor else {
            continue;
        };
        let descriptors: Vec<&LocalSet> = match descriptor.kind {
            MULTIPLE_DESCRIPTOR => descriptor
                .references(SUB_DESCRIPTORS_TAG)
                .iter()
                .filter_map(&by_uid)
                .collect(),
            _ => Vec::from([descriptor]),
        };
        let tracks: Vec<&LocalSet> = package
            .references(PACKAGE_TRACKS_TAG)
            .iter()
            .filter_map(&by_uid)
            .filter(|track| track.kind == TIMELINE_TRACK)
            .filter(|track| {
                track
                    .u32(TRACK_NUMBER_TAG)
                    .is_some_and(|number| number != 0)
            })
            .collect();
        for track in &tracks {
            let track_id = track.u32(TRACK_ID_TAG);
            let linked = descriptors
                .iter()
                .find(|descriptor| descriptor.u32(LINKED_TRACK_ID_TAG) == track_id);
            let only = (descriptors.len() == 1 && tracks.len() == 1).then(|| &descriptors[0]);
            let Some(descriptor) = linked.or(only) else {
                continue;
            };
            let (Some(track_number), Some(edit_rate)) =
                (track.u32(TRACK_NUMBER_TAG), track.rational(EDIT_RATE_TAG))
            else {
                continue;
            };
            let Some(essence) = essence(descriptor) else {
                if let Some(label) = descriptor.label(ESSENCE_CONTAINER_TAG) {
                    metadata.unsupported.push(label);
                }
                continue;
            };
            let sequence_duration = track
                .reference(TRACK_SEQUENCE_TAG)
                .and_then(|uid| by_uid(&uid))
                .filter(|sequence| sequence.kind == SEQUENCE)
                .and_then(|sequence| sequence.i64(COMPONENT_DURATION_TAG));
            let duration = sequence_duration
                .or_else(|| descriptor.i64(CONTAINER_DURATION_TAG))
                .and_then(|duration| u64::try_from(duration).ok())
                .filter(|duration| *duration > 0);
            let clip_wrapped = op_atom
                || descriptor
                    .label(ESSENCE_CONTAINER_TAG)
                    .is_some_and(|label| names_clip_wrapping(&label));
            metadata.tracks.push(EssenceTrack {
                track_number,
                edit_rate,
                duration,
                essence,
                clip_wrapped,
            });
        }
    }
    metadata
}

// the source packages this file's essence belongs to, or every source package with a descriptor
// when no essence container data names them
fn file_packages<'a, 'b>(sets: &'b [LocalSet<'a>]) -> Vec<&'b LocalSet<'a>> {
    let linked: Vec<&[u8]> = sets
        .iter()
        .filter(|set| set.kind == ESSENCE_CONTAINER_DATA)
        .filter_map(|set| set.item(LINKED_PACKAGE_UID_TAG))
        .collect();
    let source_packages = sets.iter().filter(|set| set.kind == SOURCE_PACKAGE);
    let named: Vec<&LocalSet> = source_packages
        .clone()
        .filter(|package| {
            package
                .item(PACKAGE_UID_TAG)
                .is_some_and(|uid| linked.contains(&uid))
        })
        .collect();
    if !named.is_empty() {
        return named;
    }
    source_packages
        .filter(|package| package.item(PACKAGE_DESCRIPTOR_TAG).is_some())
        .collect()
}

// generic container mappings, smpte st 379-2, by byte 13 of the essence container label
const CONTAINER_LABEL: [u8; 13] = [
    0x06, 0x0e, 0x2b, 0x34, 0x04, 0x01, 0x01, 0x01, 0x0d, 0x01, 0x03, 0x01, 0x02,
];
const D10_MAPPING: u8 = 0x01;
const MPEG_ES_MAPPING: u8 = 0x04;
const PCM_MAPPING: u8 = 0x06;
const JPEG2000_MAPPING: u8 = 0x0c;
const AVC_MAPPING: u8 = 0x10;
const VC3_MAPPING: u8 = 0x11;
const PRORES_MAPPING: u8 = 0x1c;

// byte 14 of the sound mappings: bwf and aes3, each frame then clip wrapped
const PCM_CLIP_WRAPPINGS: [u8; 2] = [0x02, 0x04];

fn names_clip_wrapping(container: &Label) -> bool {
    container[13] == PCM_MAPPING && PCM_CLIP_WRAPPINGS.contains(&container[14])
}

const PICTURE_CODING: [u8; 12] = [
    0x06, 0x0e, 0x2b, 0x34, 0x04, 0x01, 0x01, 0x01, 0x04, 0x01, 0x02, 0x02,
];
// byte 12 then 13 of a picture coding label, the h.264 profiles under mpeg compression
const AVC_CODINGS: [[u8; 2]; 2] = [[0x01, 0x31], [0x01, 0x32]];
const PRORES_CODING: [u8; 2] = [0x03, 0x06];
const MAX_CHANNELS: u32 = u8::MAX as u32;

fn essence(descriptor: &LocalSet) -> Option<Essence> {
    let container = descriptor.label(ESSENCE_CONTAINER_TAG)?;
    if !label_starts_with(&container, &CONTAINER_LABEL) {
        return None;
    }
    let mapping = container[13];
    if mapping == PCM_MAPPING {
        return pcm(descriptor);
    }
    let coding = descriptor
        .label(PICTURE_CODING_TAG)
        .filter(|coding| label_starts_with(coding, &PICTURE_CODING));
    let coding_bytes = coding.map(|coding| [coding[12], coding[13]]);
    let codec = match mapping {
        D10_MAPPING => VideoCodec::Mpeg2,
        MPEG_ES_MAPPING if coding_bytes.is_some_and(|bytes| AVC_CODINGS.contains(&bytes)) => {
            VideoCodec::H264
        }
        MPEG_ES_MAPPING => VideoCodec::Mpeg2,
        AVC_MAPPING => VideoCodec::H264,
        JPEG2000_MAPPING => VideoCodec::Jpeg2000,
        VC3_MAPPING => VideoCodec::DnxHd,
        PRORES_MAPPING => {
            let coding = coding.filter(|coding| coding[12..14] == PRORES_CODING)?;
            let variant = ProResVariant::ALL.get(usize::from(coding[14]).checked_sub(1)?)?;
            VideoCodec::ProRes { variant: *variant }
        }
        _ => return None,
    };
    if !PICTURE_DESCRIPTORS.contains(&descriptor.kind) {
        return None;
    }
    // stored sizes are padded to whole macroblocks
    let width = descriptor
        .u32(DISPLAY_WIDTH_TAG)
        .or_else(|| descriptor.u32(STORED_WIDTH_TAG))?;
    let field_height = descriptor
        .u32(DISPLAY_HEIGHT_TAG)
        .or_else(|| descriptor.u32(STORED_HEIGHT_TAG))?;
    let height = match descriptor.u8(FRAME_LAYOUT_TAG) {
        Some(SEPARATE_FIELDS | MIXED_FIELDS) => field_height.checked_mul(2)?,
        _ => field_height,
    };
    (width > 0 && height > 0).then_some(Essence::Video {
        codec,
        width,
        height,
    })
}

// smpte st 377-1 frame layouts whose stored height is one field's
const SEPARATE_FIELDS: u8 = 1;
const MIXED_FIELDS: u8 = 3;

fn pcm(descriptor: &LocalSet) -> Option<Essence> {
    if !SOUND_DESCRIPTORS.contains(&descriptor.kind) {
        return None;
    }
    let format = match descriptor.u32(QUANTIZATION_BITS_TAG)? {
        16 => AudioFormat::PcmS16Le,
        24 => AudioFormat::PcmS24Le,
        32 => AudioFormat::PcmS32Le,
        _ => return None,
    };
    let channels = descriptor.u32(CHANNEL_COUNT_TAG)?;
    if channels == 0 || channels > MAX_CHANNELS {
        return None;
    }
    let rate = descriptor.rational(AUDIO_SAMPLING_RATE_TAG)?;
    if rate.numerator % rate.denominator != 0 {
        return None;
    }
    Some(Essence::Pcm {
        format,
        channels: channels as u8,
        sample_rate: rate.numerator / rate.denominator,
    })
}

// ---- index tables ----

const INDEX_EDIT_RATE_TAG: u16 = 0x3f0b;
const INDEX_START_TAG: u16 = 0x3f0c;
const INDEX_DURATION_TAG: u16 = 0x3f0d;
const EDIT_UNIT_BYTE_COUNT_TAG: u16 = 0x3f05;
const INDEX_BODY_SID_TAG: u16 = 0x3f07;
const INDEX_ENTRY_ARRAY_TAG: u16 = 0x3f0a;
// temporal offset, key frame offset, flags and stream offset
const INDEX_ENTRY_MIN_LEN: usize = 11;
const RANDOM_ACCESS_FLAG: u8 = 0x80;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IndexEntry {
    pub temporal_offset: i8,
    pub key_frame_offset: i8,
    pub random_access: bool,
    pub stream_offset: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IndexSegment {
    pub edit_rate: Rational,
    pub start: u64,
    pub duration: u64,
    pub edit_unit_byte_count: u32,
    pub body_sid: u32,
    pub entries: Vec<IndexEntry>,
}

pub(crate) fn index_segment(value: &[u8]) -> Option<IndexSegment> {
    let set = local_set(INDEX_SEGMENT_KIND, value)?;
    let entries = set
        .item(INDEX_ENTRY_ARRAY_TAG)
        .map(index_entries)
        .unwrap_or(Some(Vec::new()))?;
    Some(IndexSegment {
        edit_rate: set.rational(INDEX_EDIT_RATE_TAG)?,
        start: u64::try_from(set.i64(INDEX_START_TAG)?).ok()?,
        duration: set
            .i64(INDEX_DURATION_TAG)
            .and_then(|duration| u64::try_from(duration).ok())
            .unwrap_or(0),
        edit_unit_byte_count: set.u32(EDIT_UNIT_BYTE_COUNT_TAG).unwrap_or(0),
        body_sid: set.u32(INDEX_BODY_SID_TAG).unwrap_or(0),
        entries,
    })
}

fn index_entries(value: &[u8]) -> Option<Vec<IndexEntry>> {
    let count = be_u32(value, 0)? as usize;
    let len = be_u32(value, 4)? as usize;
    if len < INDEX_ENTRY_MIN_LEN {
        return None;
    }
    let entries = value.get(BATCH_HEADER_LEN..)?;
    if entries.len() / len < count {
        return None;
    }
    entries
        .chunks_exact(len)
        .take(count)
        .map(|entry| {
            Some(IndexEntry {
                temporal_offset: entry[0] as i8,
                key_frame_offset: entry[1] as i8,
                random_access: entry[2] & RANDOM_ACCESS_FLAG != 0,
                stream_offset: be_u64(entry, 3)?,
            })
        })
        .collect()
}

fn skip_fill(file: &[u8], mut at: usize) -> Option<usize> {
    loop {
        let (klv, value) = whole_klv(file.get(at..)?).ok()?;
        if key_kind(&klv.key) != KeyKind::Fill {
            return Some(at);
        }
        at += klv.header_len + value.len();
    }
}

// the header partition's metadata at the start of `file`, None until all of it is in
pub(crate) fn header_partition_metadata(file: &[u8]) -> Option<HeaderMetadata> {
    let (klv, value) = whole_klv(file).ok()?;
    if !matches!(key_kind(&klv.key), KeyKind::Partition(_)) {
        return None;
    }
    let pack = partition_pack(value)?;
    // header byte count starts at the primer pack, after the fill that pads the partition pack
    let start = skip_fill(file, klv.header_len + value.len())?;
    let end = start.checked_add(usize::try_from(pack.header_byte_count).ok()?)?;
    Some(header_metadata(file.get(start..end)?, pack.op_atom))
}

pub(crate) fn opens_header_partition(file: &[u8]) -> bool {
    read_klv(file).is_ok_and(|klv| key_kind(&klv.key) == KeyKind::Partition(PartitionKind::Header))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MPEG2_PCM: &[u8] = include_bytes!("../tests/fixtures/mxf_mpeg2_pcm.mxf");
    const IMF_PCM_CLIP: &[u8] = include_bytes!("../tests/fixtures/mxf_imf_pcm_clip.mxf");
    const PCM_OPATOM: &[u8] = include_bytes!("../tests/fixtures/mxf_pcm_opatom.mxf");
    const FRAME_RATE: Rational = Rational {
        numerator: 25,
        denominator: 1,
    };
    const FIXTURE_FRAMES: u64 = 10;

    fn video(file: &[u8]) -> Essence {
        let metadata = header_partition_metadata(file).unwrap();
        let [track] = metadata.tracks.as_slice() else {
            panic!("one track expected, got {:?}", metadata.tracks);
        };
        assert_eq!(track.edit_rate, FRAME_RATE);
        assert_eq!(track.duration, Some(FIXTURE_FRAMES));
        track.essence
    }

    #[test]
    fn op1a_header_metadata_names_a_picture_and_a_sound_track() {
        let metadata = header_partition_metadata(MPEG2_PCM).unwrap();
        let essences: Vec<Essence> = metadata.tracks.iter().map(|track| track.essence).collect();
        assert_eq!(
            essences,
            [
                Essence::Video {
                    codec: VideoCodec::Mpeg2,
                    width: 64,
                    height: 48,
                },
                Essence::Pcm {
                    format: AudioFormat::PcmS24Le,
                    channels: 2,
                    sample_rate: 48_000,
                },
            ]
        );
        assert!(metadata
            .tracks
            .iter()
            .all(|track| track.edit_rate == FRAME_RATE && track.duration == Some(FIXTURE_FRAMES)));
    }

    #[test]
    fn each_picture_mapping_reads_as_its_codec() {
        let cases: [(&[u8], VideoCodec, u32, u32); 4] = [
            (
                include_bytes!("../tests/fixtures/mxf_h264.mxf"),
                VideoCodec::H264,
                64,
                48,
            ),
            (
                include_bytes!("../tests/fixtures/mxf_prores.mxf"),
                VideoCodec::ProRes {
                    variant: ProResVariant::Proxy,
                },
                64,
                48,
            ),
            (
                include_bytes!("../tests/fixtures/mxf_jpeg2000.mxf"),
                VideoCodec::Jpeg2000,
                64,
                48,
            ),
            (
                include_bytes!("../tests/fixtures/mxf_dnxhr.mxf"),
                VideoCodec::DnxHd,
                256,
                120,
            ),
        ];
        for (file, codec, width, height) in cases {
            assert_eq!(
                video(file),
                Essence::Video {
                    codec,
                    width,
                    height
                }
            );
        }
    }

    #[test]
    fn sound_files_name_their_layout() {
        let opatom = header_partition_metadata(PCM_OPATOM).unwrap();
        assert_eq!(
            opatom.tracks[0].essence,
            Essence::Pcm {
                format: AudioFormat::PcmS24Le,
                channels: 1,
                sample_rate: 48_000,
            }
        );
        let clip = header_partition_metadata(IMF_PCM_CLIP).unwrap();
        let [track] = clip.tracks.as_slice() else {
            panic!("one track expected, got {:?}", clip.tracks);
        };
        assert_eq!(
            track.essence,
            Essence::Pcm {
                format: AudioFormat::PcmS24Le,
                channels: 2,
                sample_rate: 48_000,
            }
        );
    }

    fn index_segments(file: &[u8]) -> Vec<IndexSegment> {
        let mut segments = Vec::new();
        let mut at = 0;
        while let Ok((klv, value)) = whole_klv(&file[at..]) {
            if key_kind(&klv.key) == KeyKind::IndexSegment {
                segments.push(index_segment(value).unwrap());
            }
            at += klv.header_len + value.len();
        }
        segments
    }

    #[test]
    fn index_entries_carry_the_reordering_and_the_seek_points() {
        let [segment] = index_segments(MPEG2_PCM).try_into().unwrap();
        assert_eq!(segment.edit_rate, FRAME_RATE);
        assert_eq!(segment.duration, FIXTURE_FRAMES);
        let temporal: Vec<i8> = segment
            .entries
            .iter()
            .map(|entry| entry.temporal_offset)
            .collect();
        assert_eq!(temporal, [0, 1, 1, -2, 1, 1, -2, 1, 1, -2]);
        assert!(segment.entries[0].random_access);
        assert_eq!(segment.entries[0].stream_offset, 0);
    }

    #[test]
    fn the_random_index_pack_lists_every_partition() {
        let rip_len = random_index_pack_len(IMF_PCM_CLIP).unwrap() as usize;
        let rip = &IMF_PCM_CLIP[IMF_PCM_CLIP.len() - rip_len..];
        let (klv, value) = whole_klv(rip).unwrap();
        assert_eq!(key_kind(&klv.key), KeyKind::RandomIndexPack);
        let partitions = random_index_pack(value).unwrap();
        for offset in &partitions {
            let (klv, _) = whole_klv(&IMF_PCM_CLIP[*offset as usize..]).unwrap();
            assert!(matches!(key_kind(&klv.key), KeyKind::Partition(_)));
        }
        assert_eq!(partitions.first(), Some(&0));
    }

    #[test]
    fn malformed_lengths_fail_instead_of_reading_past_the_value() {
        let mut long_ber = Vec::from(&MPEG2_PCM[..KEY_LEN]);
        long_ber.extend_from_slice(&[0x89, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(read_klv(&long_ber), Err(KlvError::Malformed));
        assert_eq!(read_klv(&MPEG2_PCM[..10]), Err(KlvError::Truncated));
        let item_past_the_end = [0x3c, 0x0a, 0x00, 0x20, 0x00];
        assert!(local_set(SOURCE_PACKAGE, &item_past_the_end).is_none());
        assert!(header_partition_metadata(&MPEG2_PCM[..4096]).is_none());
    }
}
