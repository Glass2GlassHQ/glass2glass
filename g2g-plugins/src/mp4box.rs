//! Shared fragmented-MP4 box primitives for the MP4 muxer/demuxer elements
//! (`mp4sink`/`mp4src` and their audio counterparts). Writers build
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

/// The `ftyp` box (iso5/isom brands), identical for the video and audio muxers.
pub(crate) fn ftyp() -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(b"iso5"); // major brand
    p.extend_from_slice(&512u32.to_be_bytes()); // minor version
    p.extend_from_slice(b"iso5");
    p.extend_from_slice(b"isom");
    mp4_box(b"ftyp", &p)
}

// --- readers ---------------------------------------------------------------

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
        i += size;
        Some((kind, payload))
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

/// iTunes-style metadata from `moov/udta/meta/ilst`, mapped to a [`TagList`]
/// (empty when the file has none). `meta` is a FullBox (a 4-byte version/flags
/// before its children), so its body is tried both with and without that prefix
/// for writers that omit it. Each `ilst` child is an item box named by a 4cc
/// (`©nam`, `©ART`, ...) holding a `data` box; UTF-8 text items become tags, the
/// 4cc mapped to a common key or kept verbatim in [`Tag::Other`].
pub(crate) fn parse_ilst_tags(moov: &[u8]) -> TagList {
    let mut list = TagList::new();
    let Some(udta) = find_box(moov, b"udta") else { return list };
    let Some(meta) = find_box(udta, b"meta") else { return list };
    let after_fullbox = meta.get(4..).unwrap_or(meta);
    let Some(ilst) = find_box(after_fullbox, b"ilst").or_else(|| find_box(meta, b"ilst")) else {
        return list;
    };
    for (kind, item) in boxes(ilst) {
        if let Some(value) = ilst_text(item) {
            list.push(itunes_tag(kind, &value));
        }
    }
    list
}

/// The UTF-8 text out of an item's `data` box. The data box body is
/// `[u32 type][u32 locale][value]`; well-known type 1 is UTF-8 text. `None` for a
/// non-text or malformed item.
fn ilst_text(item: &[u8]) -> Option<String> {
    let data = find_box(item, b"data")?;
    if be32(data, 0).ok()? != 1 {
        return None;
    }
    core::str::from_utf8(data.get(8..)?).ok().map(String::from)
}

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
            let key: String = kind.iter().filter(|&&b| b.is_ascii()).map(|&b| b as char).collect();
            return Tag::Other { key, value: value.into() };
        }
    };
    Tag::from_key_value(name, value)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(tags.tags(), &[Tag::Other { key: "keyw".into(), value: "rust".into() }]);
    }

    #[test]
    fn no_udta_is_empty() {
        let moov = mp4_box(b"moov", &mp4_box(b"trak", &[]));
        assert!(parse_ilst_tags(find_box(&moov, b"moov").unwrap()).is_empty());
    }
}
