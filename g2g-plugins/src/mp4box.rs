//! Shared fragmented-MP4 box primitives for the MP4 muxer/demuxer elements
//! (`fmp4mux`/`mp4src` and their audio counterparts). Writers build
//! size-prefixed boxes; readers walk the box tree. std-gated like its callers.

use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::{G2gError, Tag, TagList};

/// Unity 3x3 transform matrix (16.16 / 2.30 fixed point) for `tkhd`/`mvhd`.
pub(crate) const MATRIX: [u32; 9] = [0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000];

// --- writers ---------------------------------------------------------------

/// A size-prefixed box: `[u32 size][4cc kind][payload]`.
pub(crate) fn mp4_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(8 + payload.len());
    b.extend_from_slice(&((payload.len() as u32 + 8).to_be_bytes()));
    b.extend_from_slice(kind);
    b.extend_from_slice(payload);
    b
}

/// A full box: a version byte plus 24-bit flags, then the payload.
pub(crate) fn full_box(kind: &[u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
    let mut p = Vec::with_capacity(4 + payload.len());
    p.push(version);
    p.extend_from_slice(&flags.to_be_bytes()[1..]);
    p.extend_from_slice(payload);
    mp4_box(kind, &p)
}

/// A brand box (`ftyp` / `styp`): major brand, minor version, compatible brands.
fn brand_box(kind: &[u8; 4], major: &[u8; 4], minor: u32, compatible: &[&[u8; 4]]) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(major);
    p.extend_from_slice(&minor.to_be_bytes());
    for b in compatible {
        p.extend_from_slice(*b);
    }
    mp4_box(kind, &p)
}

/// The `ftyp` box (iso5/isom brands), identical for the video and audio muxers.
pub(crate) fn ftyp() -> Vec<u8> {
    brand_box(b"ftyp", b"iso5", 512, &[b"iso5", b"isom"])
}

/// The `ftyp` of a CMAF track file (M832). `cmfc` is CMAF's structural brand: one
/// media track, `mvex`/`trex`, a `tfdt` in every `traf`, `default-base-is-moof`,
/// and every fragment starting at a stream access point. The stricter `cmf2` is
/// deliberately not claimed: its "sample defaults repeated in each track
/// fragment" rule is not something this writer's per-sample `trun` demonstrably
/// satisfies, and ffmpeg / shaka-packager declare only `cmfc` too. `iso6` covers
/// the `tfdt` a plain ISO-BMFF reader needs to know about.
pub(crate) fn ftyp_cmaf() -> Vec<u8> {
    brand_box(b"ftyp", b"cmfc", 0, &[b"cmfc", b"iso6", b"isom", b"mp41"])
}

/// The `styp` that opens each CMAF segment (M832), so one fragment of a track
/// file is separately addressable by an HLS `#EXT-X-BYTERANGE` or a DASH
/// `SegmentBase` range. Each unit this muxer emits is one CMAF fragment which is
/// also one CMAF segment, hence both `cmfs` and `cmff`; `msdh` marks it a generic
/// DASH media segment. `chunked` adds the CMAF chunk brand `cmfl` (M859): the
/// segment is then delivered as a sequence of `moof`+`mdat` chunks, each a
/// separately addressable prefix of it.
pub(crate) fn styp_cmaf(chunked: bool) -> Vec<u8> {
    let compatible: &[&[u8; 4]] = if chunked {
        &[b"cmfs", b"cmff", b"cmfl", b"msdh"]
    } else {
        &[b"cmfs", b"cmff", b"msdh"]
    };
    brand_box(b"styp", b"cmfs", 0, compatible)
}

/// The `prft` (ProducerReferenceTimeBox, ISO/IEC 14496-12 8.16.5) written ahead
/// of a fragment's `moof` (M859): it maps `media_time`, the decode time of the
/// fragment's first sample in the track's timescale, to the producer's wall clock
/// as a 64-bit NTP timestamp, which is how a low-latency player measures its own
/// end-to-end latency. Version 1 (64-bit `media_time`), matching what ffmpeg's
/// `-write_prft wallclock` writes. Flags stay `0`, which DASH-IF maps to
/// `ProducerReferenceTime@type = encoder`: the time is read where the samples are
/// muxed, not carried from a capture device.
pub(crate) fn prft(reference_track_id: u32, ntp: u64, media_time: u64) -> Vec<u8> {
    let mut p = Vec::with_capacity(20);
    p.extend_from_slice(&reference_track_id.to_be_bytes());
    p.extend_from_slice(&ntp.to_be_bytes());
    p.extend_from_slice(&media_time.to_be_bytes());
    full_box(b"prft", 1, 0, &p)
}

// --- readers ---------------------------------------------------------------

/// Total length of the box at the start of `buf`. `Ok(None)` means the 8-byte
/// header (or the 64-bit large-size header) isn't fully buffered yet. Once the
/// size field is in hand, a value below 8 (including the size-0 "to end of
/// stream" form) is malformed and fails loud rather than stalling a streaming
/// reader with an unconsumable box.
pub(crate) fn next_box_len(buf: &[u8]) -> Result<Option<usize>, G2gError> {
    if buf.len() < 8 {
        return Ok(None);
    }
    let size = u32::from_be_bytes(buf[0..4].try_into().expect("4 bytes"));
    let total = if size == 1 {
        if buf.len() < 16 {
            return Ok(None);
        }
        u64::from_be_bytes(buf[8..16].try_into().expect("8 bytes")) as usize
    } else {
        size as usize
    };
    if total < 8 {
        return Err(G2gError::CapsMismatch);
    }
    Ok(Some(total))
}

pub(crate) fn be32(data: &[u8], at: usize) -> Result<u32, G2gError> {
    data.get(at..at + 4)
        .map(|b| u32::from_be_bytes(b.try_into().expect("4 bytes")))
        .ok_or(G2gError::CapsMismatch)
}

pub(crate) fn be64(data: &[u8], at: usize) -> Result<u64, G2gError> {
    data.get(at..at + 8)
        .map(|b| u64::from_be_bytes(b.try_into().expect("8 bytes")))
        .ok_or(G2gError::CapsMismatch)
}

/// Iterate the child boxes of `data`, yielding `(fourcc, payload)`.
pub(crate) fn boxes(data: &[u8]) -> impl Iterator<Item = (&[u8; 4], &[u8])> {
    boxes_at(data).map(|(kind, payload, _)| (kind, payload))
}

/// [`boxes`] plus each box's start offset in `data`, for the box whose contents
/// are addressed by offsets relative to its own header (`saio` inside a `moof`).
pub(crate) fn boxes_at(data: &[u8]) -> impl Iterator<Item = (&[u8; 4], &[u8], usize)> {
    let mut i = 0usize;
    core::iter::from_fn(move || {
        if i + 8 > data.len() {
            return None;
        }
        let size = u32::from_be_bytes(data[i..i + 4].try_into().expect("4 bytes")) as usize;
        if size < 8 || i + size > data.len() {
            return None;
        }
        let kind: &[u8; 4] = data[i + 4..i + 8].try_into().expect("4 bytes");
        let payload = &data[i + 8..i + size];
        let start = i;
        i += size;
        Some((kind, payload, start))
    })
}

pub(crate) fn find_box<'a>(data: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    boxes(data).find(|(k, _)| *k == kind).map(|(_, p)| p)
}

/// Descend a path of nested boxes.
pub(crate) fn find_path<'a>(mut data: &'a [u8], path: &[&[u8; 4]]) -> Option<&'a [u8]> {
    for kind in path {
        data = find_box(data, kind)?;
    }
    Some(data)
}

/// Extract the AudioSpecificConfig from an `esds` payload by descending the
/// descriptor tree: ES (0x03) -> DecoderConfig (0x04) -> DecoderSpecific (0x05).
/// Shared by the audio MP4 source and the multi-track fMP4 parser.
pub(crate) fn parse_esds(esds: &[u8]) -> Result<Vec<u8>, G2gError> {
    // skip the full-box version/flags (4 bytes).
    let es = find_descriptor(esds.get(4..).ok_or(G2gError::CapsMismatch)?, 0x03)
        .ok_or(G2gError::CapsMismatch)?;
    // ES_Descriptor payload: ES_ID (2) + flags (1), then sub-descriptors.
    let dcd = find_descriptor(es.get(3..).ok_or(G2gError::CapsMismatch)?, 0x04)
        .ok_or(G2gError::CapsMismatch)?;
    // DecoderConfigDescriptor: 13 fixed bytes, then DecoderSpecificInfo.
    let asc = find_descriptor(dcd.get(13..).ok_or(G2gError::CapsMismatch)?, 0x05)
        .ok_or(G2gError::CapsMismatch)?;
    if asc.is_empty() {
        return Err(G2gError::CapsMismatch);
    }
    Ok(asc.to_vec())
}

/// Extract the objectTypeIndication and DecoderSpecificInfo from a video `esds`
/// (the `mp4v` sample entry's config), descending ES (0x03) -> DecoderConfig
/// (0x04). The objectTypeIndication is the first of the DecoderConfig's 13 fixed
/// bytes (`0x20` = Visual ISO/IEC 14496-2, i.e. MPEG-4 Part 2); the caller uses
/// it to confirm the codec before tagging. The DecoderSpecificInfo (0x05) is the
/// VOL header the software decoder wants as in-band config; some muxers omit it
/// and carry the config in-band, so its absence is tolerated (empty vec).
pub(crate) fn parse_esds_video(esds: &[u8]) -> Result<(u8, Vec<u8>), G2gError> {
    // skip the full-box version/flags (4 bytes).
    let es = find_descriptor(esds.get(4..).ok_or(G2gError::CapsMismatch)?, 0x03)
        .ok_or(G2gError::CapsMismatch)?;
    // ES_Descriptor payload: ES_ID (2) + flags (1), then sub-descriptors.
    let dcd = find_descriptor(es.get(3..).ok_or(G2gError::CapsMismatch)?, 0x04)
        .ok_or(G2gError::CapsMismatch)?;
    let oti = *dcd.first().ok_or(G2gError::CapsMismatch)?;
    // DecoderConfigDescriptor: 13 fixed bytes, then the optional DecoderSpecificInfo.
    let dsi = dcd
        .get(13..)
        .and_then(|rest| find_descriptor(rest, 0x05))
        .map(<[u8]>::to_vec)
        .unwrap_or_default();
    Ok((oti, dsi))
}

/// Find the first descriptor with `tag` among the descriptors laid out at the
/// start of `data`, returning its payload. Handles the expandable size encoding
/// (7 bits per byte, high bit a continuation flag).
pub(crate) fn find_descriptor(data: &[u8], tag: u8) -> Option<&[u8]> {
    let mut i = 0usize;
    while i < data.len() {
        let t = data[i];
        i += 1;
        let mut size = 0usize;
        loop {
            let b = *data.get(i)?;
            i += 1;
            size = (size << 7) | (b & 0x7F) as usize;
            if b & 0x80 == 0 {
                break;
            }
        }
        let payload = data.get(i..i + size)?;
        if t == tag {
            return Some(payload);
        }
        i += size;
    }
    None
}

/// iTunes-style metadata from a container's `udta/meta/ilst`, mapped to a
/// [`TagList`] (empty when it has none). The container is a `moov` for the
/// file's own tags or a `trak` for one track's (M838). `meta` is a FullBox (a
/// 4-byte version/flags before its children), so its body is tried both with and
/// without that prefix for writers that omit it. Each `ilst` child is an item
/// box named by a 4cc (`©nam`, `©ART`, ...) holding a `data` box: UTF-8 text
/// items become tags with the 4cc mapped to a common key or kept verbatim in
/// [`Tag::Other`], the integer atoms become [`Tag::Number`]s, and a `----` item
/// becomes a [`Tag::Freeform`] under its `mean` namespace.
pub(crate) fn parse_ilst_tags(container: &[u8]) -> TagList {
    let mut list = TagList::new();
    let Some(udta) = find_box(container, b"udta") else {
        return list;
    };
    let Some(meta) = find_box(udta, b"meta") else {
        return list;
    };
    let after_fullbox = meta.get(4..).unwrap_or(meta);
    let Some(ilst) = find_box(after_fullbox, b"ilst").or_else(|| find_box(meta, b"ilst")) else {
        return list;
    };
    for (kind, item) in boxes(ilst) {
        if kind == b"----" {
            if let Some(tag) = freeform_tag(item) {
                list.push(tag);
            }
        } else if let Some(numbers) = ilst_numbers(kind, item) {
            for tag in numbers {
                list.push(tag);
            }
        } else if let Some(value) = ilst_text(item) {
            list.push(itunes_tag(kind, &value));
        }
    }
    list
}

/// An item's `data` box as `(well-known type, value bytes)`. The body is
/// `[u32 type][u32 locale][value]`.
fn ilst_data(item: &[u8]) -> Option<(u32, &[u8])> {
    let data = find_box(item, b"data")?;
    Some((be32(data, 0).ok()?, data.get(8..)?))
}

/// The UTF-8 text out of an item's `data` box (well-known type 1). `None` for a
/// non-text or malformed item.
fn ilst_text(item: &[u8]) -> Option<String> {
    let (kind, value) = ilst_data(item)?;
    if kind != 1 {
        return None;
    }
    core::str::from_utf8(value).ok().map(String::from)
}

/// The integer iTunes atoms, as [`Tag::Number`]s. `trkn` / `disk` carry an
/// index and a total in one implicit-type (0) payload
/// `[u16 reserved][u16 index][u16 total][u16 reserved]`, so they yield up to two
/// tags (a zero total means "unknown", and is dropped). `tmpo` / `cpil` carry
/// one big-endian integer (well-known type 21). `None` for any other atom, so
/// the caller falls through to the text path.
fn ilst_numbers(kind: &[u8; 4], item: &[u8]) -> Option<Vec<Tag>> {
    let (_data_type, value) = ilst_data(item)?;
    let number = |key: &str, v: u64| Tag::Number {
        key: String::from(key),
        value: v,
    };
    if let Some((_, index_key, count_key)) =
        INT_PAIR_ATOMS.iter().find(|(atom, _, _)| kind == *atom)
    {
        let be16 = |at: usize| -> Option<u64> {
            let b = value.get(at..at + 2)?;
            Some(u64::from(u16::from_be_bytes([b[0], b[1]])))
        };
        let mut out = Vec::new();
        out.push(number(index_key, be16(2)?));
        match be16(4) {
            Some(total) if total != 0 => out.push(number(count_key, total)),
            _ => {}
        }
        return Some(out);
    }
    let (_, key, _) = INT_ATOMS.iter().find(|(atom, _, _)| kind == *atom)?;
    Some(alloc::vec![number(key, be_int(value)?)])
}

/// A big-endian integer of 1 to 8 bytes (what a well-known-type-21 `data` box
/// holds). `None` for an empty or over-wide payload.
fn be_int(value: &[u8]) -> Option<u64> {
    if value.is_empty() || value.len() > 8 {
        return None;
    }
    Some(value.iter().fold(0u64, |acc, b| (acc << 8) | u64::from(*b)))
}

/// A freeform (`----`) item as a [`Tag::Freeform`]: a `mean` full box (the
/// reverse-DNS namespace, e.g. `com.apple.iTunes`), a `name` full box (the key)
/// and a UTF-8 `data` box. `None` when any part is missing or not UTF-8.
fn freeform_tag(item: &[u8]) -> Option<Tag> {
    let text = |b: &[u8]| core::str::from_utf8(b.get(4..)?).ok().map(String::from);
    let namespace = text(find_box(item, b"mean")?)?;
    let key = text(find_box(item, b"name")?)?;
    Some(Tag::Freeform {
        namespace,
        key,
        value: ilst_text(item)?,
    })
}

/// The index/total integer atoms and the tag keys the two halves carry.
const INT_PAIR_ATOMS: &[(&[u8; 4], &str, &str)] = &[
    (b"trkn", Tag::TRACK_NUMBER, Tag::TRACK_COUNT),
    (b"disk", Tag::DISC_NUMBER, Tag::DISC_COUNT),
];

/// The single-value integer atoms: their tag key and the payload width iTunes
/// writes (`tmpo` a u16, `cpil` a u8 flag).
const INT_ATOMS: &[(&[u8; 4], &str, usize)] = &[
    (b"tmpo", Tag::BEATS_PER_MINUTE, 2),
    (b"cpil", Tag::COMPILATION, 1),
];

/// Map an iTunes metadata 4cc to a tag. The `©`-prefixed (0xA9) atoms are the
/// common text keys; an unrecognized 4cc keeps its readable name in
/// [`Tag::Other`].
fn itunes_tag(kind: &[u8; 4], value: &str) -> Tag {
    let name = match kind {
        b"\xA9nam" => "title",
        b"\xA9ART" => "artist",
        b"\xA9alb" => "album",
        b"\xA9too" => "encoder",
        b"\xA9cmt" => "comment",
        _ => {
            // strip the non-ASCII © so a stray atom keeps a printable key.
            let key: String = kind
                .iter()
                .filter(|&&b| b.is_ascii())
                .map(|&b| b as char)
                .collect();
            return Tag::Other {
                key,
                value: value.into(),
            };
        }
    };
    Tag::from_key_value(name, value)
}

/// Build a `udta/meta/ilst` box carrying `tags` (the inverse of
/// [`parse_ilst_tags`]), or `None` when none of them map to an iTunes atom. The
/// `meta` box names the `mdir` handler; a typed tag writes its `©`-prefixed text
/// atom, a [`Tag::Number`] its integer atom, and a [`Tag::Freeform`] a `----`
/// item. `Tag::Language` (an MP4 track field) and `Tag::Other` (no namespace to
/// write a `----` under) are skipped.
pub(crate) fn udta_with_tags(tags: &TagList) -> Option<Vec<u8>> {
    let ilst = ilst_items(tags);
    if ilst.is_empty() {
        return None;
    }
    let meta_body = [meta_hdlr(), mp4_box(b"ilst", &ilst)].concat();
    Some(mp4_box(b"udta", &full_box(b"meta", 0, 0, &meta_body)))
}

/// The `ilst` children for `tags`, in tag order. The two halves of an index/total
/// atom (`track-number` + `track-count`) collapse into the one atom that carries
/// both, written where the first of them appears.
fn ilst_items(tags: &TagList) -> Vec<u8> {
    let number_of = |wanted: &str| {
        tags.tags().iter().find_map(|t| match t {
            Tag::Number { key, value } if key == wanted => Some(*value),
            _ => None,
        })
    };
    let mut out = Vec::new();
    let mut pairs_written: Vec<&[u8; 4]> = Vec::new();
    for t in tags.tags() {
        match t {
            Tag::Freeform {
                namespace,
                key,
                value,
            } => out.extend_from_slice(&freeform_item(namespace, key, value)),
            Tag::Number { key, value } => {
                if let Some((atom, index_key, count_key)) = INT_PAIR_ATOMS
                    .iter()
                    .find(|(_, index, count)| key == index || key == count)
                {
                    if !pairs_written.contains(atom) {
                        pairs_written.push(atom);
                        out.extend_from_slice(&int_pair_item(
                            atom,
                            number_of(index_key).unwrap_or(0),
                            number_of(count_key).unwrap_or(0),
                        ));
                    }
                } else if let Some((atom, _, width)) = INT_ATOMS.iter().find(|(_, k, _)| key == k) {
                    out.extend_from_slice(&int_item(atom, *value, *width));
                }
                // any other integer key has no atom: dropped, like `Tag::Other`.
            }
            _ => {
                if let Some((atom, value)) = itunes_atom(t) {
                    out.extend_from_slice(&ilst_text_item(atom, value));
                }
            }
        }
    }
    out
}

/// An iTunes item box: a `©`-prefixed atom wrapping a UTF-8 (`type 1`) `data` box.
fn ilst_text_item(atom: &[u8; 4], value: &str) -> Vec<u8> {
    mp4_box(atom, &text_data_box(value))
}

/// A UTF-8 (well-known type 1) `data` box: `[u32 type][u32 locale][value]`.
fn text_data_box(value: &str) -> Vec<u8> {
    let mut data = 1u32.to_be_bytes().to_vec();
    data.extend_from_slice(&0u32.to_be_bytes()); // locale
    data.extend_from_slice(value.as_bytes());
    mp4_box(b"data", &data)
}

/// A freeform item: `----` wrapping `mean` (the namespace), `name` (the key) and
/// a UTF-8 `data` box.
fn freeform_item(namespace: &str, key: &str, value: &str) -> Vec<u8> {
    let body = [
        full_box(b"mean", 0, 0, namespace.as_bytes()),
        full_box(b"name", 0, 0, key.as_bytes()),
        text_data_box(value),
    ]
    .concat();
    mp4_box(b"----", &body)
}

/// A `trkn` / `disk` item: the implicit-type (0) index+total payload iTunes and
/// ffmpeg write. Values wider than the 16-bit fields saturate.
fn int_pair_item(atom: &[u8; 4], index: u64, total: u64) -> Vec<u8> {
    let half = |v: u64| (v.min(u64::from(u16::MAX)) as u16).to_be_bytes();
    let mut data = 0u32.to_be_bytes().to_vec(); // well-known type 0 = implicit
    data.extend_from_slice(&0u32.to_be_bytes()); // locale
    data.extend_from_slice(&0u16.to_be_bytes()); // reserved
    data.extend_from_slice(&half(index));
    data.extend_from_slice(&half(total));
    data.extend_from_slice(&0u16.to_be_bytes()); // reserved
    mp4_box(atom, &mp4_box(b"data", &data))
}

/// A single-value integer item (`tmpo` / `cpil`): well-known type 21, the value
/// big-endian in `width` bytes (saturating).
fn int_item(atom: &[u8; 4], value: u64, width: usize) -> Vec<u8> {
    let max = if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    };
    let bytes = value.min(max).to_be_bytes();
    let mut data = 21u32.to_be_bytes().to_vec(); // well-known type 21 = BE integer
    data.extend_from_slice(&0u32.to_be_bytes()); // locale
    data.extend_from_slice(&bytes[8 - width..]);
    mp4_box(atom, &mp4_box(b"data", &data))
}

/// The metadata handler box naming the `mdir` (iTunes) handler that an `ilst`
/// lives under, with the `appl` manufacturer iTunes writes.
fn meta_hdlr() -> Vec<u8> {
    let mut p = 0u32.to_be_bytes().to_vec(); // pre_defined
    p.extend_from_slice(b"mdir"); // handler_type
    p.extend_from_slice(b"appl"); // reserved[0]: manufacturer
    p.extend_from_slice(&[0u8; 8]); // reserved[1..3]
    p.push(0); // empty name (null-terminated)
    full_box(b"hdlr", 0, 0, &p)
}

/// The iTunes `©`-prefixed atom and value for a text tag, or `None` when the tag
/// has no such atom (`Language` is an MP4 track field; `Other` has no namespace
/// to write a `----` under; `Number` / `Freeform` have their own item forms).
fn itunes_atom(tag: &Tag) -> Option<(&'static [u8; 4], &str)> {
    let pair: (&'static [u8; 4], &str) = match tag {
        Tag::Title(v) => (b"\xA9nam", v),
        Tag::Artist(v) => (b"\xA9ART", v),
        Tag::Album(v) => (b"\xA9alb", v),
        Tag::Encoder(v) => (b"\xA9too", v),
        Tag::Comment(v) => (b"\xA9cmt", v),
        Tag::Language(_) | Tag::Other { .. } | Tag::Number { .. } | Tag::Freeform { .. } => {
            return None
        }
    };
    Some(pair)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `parse_esds_video` returns the DecoderConfig objectTypeIndication and the
    /// DecoderSpecificInfo; a missing DSI yields an empty vec (config in-band).
    #[test]
    fn parse_esds_video_extracts_oti_and_dsi() {
        let descriptor = |tag: u8, body: &[u8]| {
            let mut v = alloc::vec![tag, body.len() as u8];
            v.extend_from_slice(body);
            v
        };
        let build = |oti: u8, dsi: Option<&[u8]>| {
            let mut dcd_body = alloc::vec![0u8; 13];
            dcd_body[0] = oti;
            if let Some(dsi) = dsi {
                dcd_body.extend_from_slice(&descriptor(0x05, dsi));
            }
            let dcd = descriptor(0x04, &dcd_body);
            let mut es_body = alloc::vec![0u8; 3];
            es_body.extend_from_slice(&dcd);
            // The esds box payload as `find_box` yields it: 4-byte version/flags
            // then the ES_Descriptor (0x03).
            let mut payload = alloc::vec![0u8; 4];
            payload.extend_from_slice(&descriptor(0x03, &es_body));
            payload
        };

        let vol: &[u8] = &[0x00, 0x00, 0x01, 0xB0, 0x08];
        let (oti, dsi) = parse_esds_video(&build(0x20, Some(vol))).expect("parse");
        assert_eq!(oti, 0x20);
        assert_eq!(dsi, vol);

        let (oti, dsi) = parse_esds_video(&build(0x20, None)).expect("parse without DSI");
        assert_eq!(oti, 0x20);
        assert!(dsi.is_empty(), "absent DecoderSpecificInfo is tolerated");
    }

    /// An iTunes item box: a 4cc atom wrapping a UTF-8 `data` box.
    fn text_item(kind: &[u8; 4], value: &str) -> Vec<u8> {
        let mut data = 1u32.to_be_bytes().to_vec(); // type 1 = UTF-8
        data.extend_from_slice(&0u32.to_be_bytes()); // locale
        data.extend_from_slice(value.as_bytes());
        mp4_box(kind, &mp4_box(b"data", &data))
    }

    /// A `moov` whose `udta/meta/ilst` carries `items`. `meta` is a full box.
    fn moov_with_tags(items: &[Vec<u8>]) -> Vec<u8> {
        let ilst = mp4_box(b"ilst", &items.concat());
        let meta = full_box(b"meta", 0, 0, &ilst);
        let udta = mp4_box(b"udta", &meta);
        mp4_box(b"moov", &udta)
    }

    #[test]
    fn descriptor_reader_handles_single_byte_sizes() {
        // tag 0x05 with a 2-byte payload preceded by a 0x04 wrapper.
        let inner = [0x05u8, 2, 0xAA, 0xBB];
        let outer = {
            let mut v = alloc::vec![0x04u8, inner.len() as u8];
            v.extend_from_slice(&inner);
            v
        };
        let dcd = find_descriptor(&outer, 0x04).unwrap();
        let asc = find_descriptor(dcd, 0x05).unwrap();
        assert_eq!(asc, &[0xAA, 0xBB]);
    }

    #[test]
    fn descriptor_reader_handles_expandable_sizes() {
        // size 130 encoded as 0x81 0x02 (continuation).
        let mut payload = alloc::vec![0u8; 130];
        payload[0] = 1;
        let mut d = alloc::vec![0x03u8, 0x81, 0x02];
        d.extend_from_slice(&payload);
        let got = find_descriptor(&d, 0x03).unwrap();
        assert_eq!(got.len(), 130);
        assert_eq!(got[0], 1);
    }

    #[test]
    fn reads_itunes_text_tags() {
        let moov = moov_with_tags(&[
            text_item(b"\xA9nam", "My Song"),
            text_item(b"\xA9ART", "The Band"),
            text_item(b"\xA9too", "g2g"),
        ]);
        let tags = parse_ilst_tags(find_box(&moov, b"moov").unwrap());
        assert_eq!(
            tags.tags(),
            &[
                Tag::Title("My Song".into()),
                Tag::Artist("The Band".into()),
                Tag::Encoder("g2g".into()),
            ]
        );
    }

    #[test]
    fn skips_non_text_items_and_unknown_atoms() {
        // a binary cover-art item (type 13 = JPEG) is dropped; an unknown text
        // atom keeps its 4cc as the key.
        let mut cover = 13u32.to_be_bytes().to_vec();
        cover.extend_from_slice(&0u32.to_be_bytes());
        cover.extend_from_slice(&[0xFF, 0xD8, 0xFF]);
        let covr = mp4_box(b"covr", &mp4_box(b"data", &cover));
        let moov = moov_with_tags(&[covr, text_item(b"keyw", "rust")]);
        let tags = parse_ilst_tags(find_box(&moov, b"moov").unwrap());
        assert_eq!(
            tags.tags(),
            &[Tag::Other {
                key: "keyw".into(),
                value: "rust".into()
            }]
        );
    }

    /// The freeform (`----`) and integer atoms, in the byte layout ffmpeg writes:
    /// `trkn` / `disk` as an implicit-type index+total pair, `cpil` as a
    /// well-known-type-21 flag.
    #[test]
    fn reads_freeform_and_integer_atoms() {
        let data = |dtype: u32, body: &[u8]| {
            let mut d = dtype.to_be_bytes().to_vec();
            d.extend_from_slice(&0u32.to_be_bytes());
            d.extend_from_slice(body);
            mp4_box(b"data", &d)
        };
        let freeform = {
            let body = [
                full_box(b"mean", 0, 0, b"com.apple.iTunes"),
                full_box(b"name", 0, 0, b"MOOD"),
                data(1, b"calm"),
            ]
            .concat();
            mp4_box(b"----", &body)
        };
        let trkn = mp4_box(b"trkn", &data(0, &[0, 0, 0, 3, 0, 12, 0, 0]));
        // a zero total is "unknown": only the index becomes a tag.
        let disk = mp4_box(b"disk", &data(0, &[0, 0, 0, 1, 0, 0, 0, 0]));
        let cpil = mp4_box(b"cpil", &data(21, &[1]));
        let tmpo = mp4_box(b"tmpo", &data(21, &[0, 128]));
        let moov = moov_with_tags(&[freeform, trkn, disk, cpil, tmpo]);

        let tags = parse_ilst_tags(find_box(&moov, b"moov").unwrap());
        let number = |key: &str, value: u64| Tag::Number {
            key: key.into(),
            value,
        };
        assert_eq!(
            tags.tags(),
            &[
                Tag::Freeform {
                    namespace: "com.apple.iTunes".into(),
                    key: "MOOD".into(),
                    value: "calm".into()
                },
                number(Tag::TRACK_NUMBER, 3),
                number(Tag::TRACK_COUNT, 12),
                number(Tag::DISC_NUMBER, 1),
                number(Tag::COMPILATION, 1),
                number(Tag::BEATS_PER_MINUTE, 128),
            ]
        );
    }

    #[test]
    fn writer_round_trips_freeform_and_integer_atoms() {
        let tags: TagList = [
            Tag::Freeform {
                namespace: "com.g2g".into(),
                key: "SCENE".into(),
                value: "42".into(),
            },
            Tag::Number {
                key: Tag::TRACK_NUMBER.into(),
                value: 3,
            },
            Tag::Number {
                key: Tag::TRACK_COUNT.into(),
                value: 12,
            },
            Tag::Number {
                key: Tag::COMPILATION.into(),
                value: 1,
            },
        ]
        .into_iter()
        .collect();
        let moov = mp4_box(b"moov", &udta_with_tags(&tags).expect("mappable tags"));
        let read = parse_ilst_tags(find_box(&moov, b"moov").unwrap());
        assert_eq!(
            read.tags(),
            tags.tags(),
            "both halves of the pair ride one trkn atom and come back in order"
        );
        // The index/total pair is one atom, not one per half.
        let ilst = find_path(
            find_box(&moov, b"moov").unwrap(),
            &[b"udta", b"meta", b"ilst"],
        );
        let ilst = ilst.or_else(|| {
            let meta = find_path(find_box(&moov, b"moov").unwrap(), &[b"udta", b"meta"])?;
            find_box(&meta[4..], b"ilst")
        });
        let kinds: Vec<[u8; 4]> = boxes(ilst.expect("ilst")).map(|(k, _)| *k).collect();
        assert_eq!(kinds, &[*b"----", *b"trkn", *b"cpil"]);
    }

    #[test]
    fn no_udta_is_empty() {
        let moov = mp4_box(b"moov", &mp4_box(b"trak", &[]));
        assert!(parse_ilst_tags(find_box(&moov, b"moov").unwrap()).is_empty());
    }

    #[test]
    fn udta_writer_round_trips_through_the_reader() {
        let tags: TagList = [
            Tag::Title("My Song".into()),
            Tag::Encoder("g2g".into()),
            Tag::Language("eng".into()), // dropped (no atom)
            Tag::Other {
                key: "x".into(),
                value: "y".into(),
            }, // dropped (freeform)
        ]
        .into_iter()
        .collect();
        let udta = udta_with_tags(&tags).expect("mappable tags present");
        // The reader recovers only the atom-mapped tags, in order.
        let moov = mp4_box(b"moov", &udta);
        let read = parse_ilst_tags(find_box(&moov, b"moov").unwrap());
        assert_eq!(
            read.tags(),
            &[Tag::Title("My Song".into()), Tag::Encoder("g2g".into())]
        );
    }

    #[test]
    fn udta_writer_none_without_mappable_tags() {
        let tags: TagList = [Tag::Other {
            key: "x".into(),
            value: "y".into(),
        }]
        .into_iter()
        .collect();
        assert!(udta_with_tags(&tags).is_none());
    }
}
