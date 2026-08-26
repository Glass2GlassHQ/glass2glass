//! ID3 metadata parsing, shared by [`crate::id3demux`] and
//! [`crate::mpegaudioparse`].
//!
//! An MPEG audio elementary stream carries its metadata outside the audio: an
//! ID3v2 tag ahead of the first frame, an ID3v1 128-byte block after the last.
//! Neither is audio, so a parser must skip both, and both are worth surfacing as
//! a [`TagList`] on the bus.
//!
//! Every length in a tag is read from the stream and is therefore
//! attacker-controlled: sizes are checked against what actually arrived before
//! any slice, and a frame whose declared size does not fit ends the walk rather
//! than truncating it into the next frame's bytes.

use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::{Tag, TagList};

/// The three bytes an ID3v2 tag opens with.
const ID3V2_MAGIC: [u8; 3] = *b"ID3";
/// ID3v2 header: magic, major version, revision, flags, 4 syncsafe size bytes.
/// A stream's head has to hold this much before it can be told from audio.
pub(crate) const ID3V2_HEADER_LEN: usize = 10;
/// Header flag: a copy of the header is appended after the tag body.
const ID3V2_FLAG_FOOTER: u8 = 0x10;
/// Header flag: an extended header precedes the frames.
const ID3V2_FLAG_EXTENDED: u8 = 0x40;
/// Header flag: the whole tag body is unsynchronised (0xFF 00 byte stuffing).
const ID3V2_FLAG_UNSYNC: u8 = 0x80;
/// A syncsafe integer carries 7 bits per byte, the 8th always clear.
const SYNCSAFE_BITS: u32 = 7;
/// The ID3v2 major versions whose frames this parses: 2.3 and 2.4.
const ID3V2_VERSION_2_3: u8 = 3;
const ID3V2_VERSION_2_4: u8 = 4;
/// Frame header of ID3v2.3 / 2.4: 4-byte id, 4-byte size, 2 flag bytes.
const ID3V2_FRAME_HEADER_LEN: usize = 10;
/// ID3v2.3 frame flags (second flag byte): compressed, encrypted, grouped.
const ID3V2_3_FRAME_UNREADABLE: u8 = 0xC0;
/// ID3v2.4 frame flags (second flag byte): unsynchronised, compressed,
/// encrypted. A data-length indicator (0x01) rides with those, so its bit does
/// not have to be tested separately.
const ID3V2_4_FRAME_UNREADABLE: u8 = 0x0E;

/// Text-encoding byte leading every ID3v2 text frame.
const ENCODING_LATIN1: u8 = 0;
const ENCODING_UTF16_BOM: u8 = 1;
const ENCODING_UTF16BE: u8 = 2;
const ENCODING_UTF8: u8 = 3;

/// The three bytes an ID3v1 block opens with.
const ID3V1_MAGIC: [u8; 3] = *b"TAG";
/// An ID3v1 block is exactly this long, always at the end of the file.
pub(crate) const ID3V1_LEN: usize = 128;
/// ID3v1 field widths: title, artist and album are 30 bytes each, the year 4.
const ID3V1_TEXT_LEN: usize = 30;
const ID3V1_YEAR_LEN: usize = 4;

/// Total byte length of the ID3v2 tag `buf` opens with (header, body and the
/// footer when one is declared), or `None` when it does not open on one or is
/// too short to carry a header. The size is a syncsafe 28-bit integer, so the
/// result cannot overflow a `usize` on any target this builds for.
pub(crate) fn id3v2_len(buf: &[u8]) -> Option<usize> {
    let header = buf.get(..ID3V2_HEADER_LEN)?;
    if header[..ID3V2_MAGIC.len()] != ID3V2_MAGIC {
        return None;
    }
    // 0xFF is invalid in both version bytes, the cheapest guard against a
    // "ID3"-looking byte run in audio data.
    if header[3] == 0xFF || header[4] == 0xFF {
        return None;
    }
    let size = syncsafe_u32(&header[6..ID3V2_HEADER_LEN])? as usize;
    let footer = if header[5] & ID3V2_FLAG_FOOTER != 0 {
        ID3V2_HEADER_LEN
    } else {
        0
    };
    Some(ID3V2_HEADER_LEN + size + footer)
}

/// Decode a 4-byte syncsafe integer, or `None` if any byte has its top bit set
/// (which no valid syncsafe integer does).
fn syncsafe_u32(bytes: &[u8]) -> Option<u32> {
    let mut value = 0u32;
    for &b in bytes.get(..4)? {
        if b & 0x80 != 0 {
            return None;
        }
        value = (value << SYNCSAFE_BITS) | u32::from(b);
    }
    Some(value)
}

/// Parse the text frames of the complete ID3v2 tag `tag` opens with (the whole
/// tag, header included, as [`id3v2_len`] measures it). An empty list is the
/// answer for everything this deliberately does not read: an ID3v2.2 tag (3-byte
/// frame ids), a whole-tag unsynchronised body, and any frame that is
/// compressed, encrypted or individually unsynchronised.
pub(crate) fn parse_id3v2(tag: &[u8]) -> TagList {
    let mut tags = TagList::new();
    let Some(header) = tag.get(..ID3V2_HEADER_LEN) else {
        return tags;
    };
    let version = header[3];
    if version != ID3V2_VERSION_2_3 && version != ID3V2_VERSION_2_4 {
        return tags;
    }
    if header[5] & ID3V2_FLAG_UNSYNC != 0 {
        return tags;
    }
    let Some(size) = syncsafe_u32(&header[6..ID3V2_HEADER_LEN]) else {
        return tags;
    };
    // The body is whatever of the declared size actually arrived.
    let end = (ID3V2_HEADER_LEN + size as usize).min(tag.len());
    let Some(body) = tag.get(ID3V2_HEADER_LEN..end) else {
        return tags;
    };
    let mut at = if header[5] & ID3V2_FLAG_EXTENDED != 0 {
        match extended_header_len(body, version) {
            Some(len) => len,
            None => return tags,
        }
    } else {
        0
    };
    while let Some(frame_header) = body.get(at..at + ID3V2_FRAME_HEADER_LEN) {
        // Padding after the last frame is zero bytes, so a zero id ends the walk.
        if frame_header[0] == 0 {
            break;
        }
        let size = frame_size(&frame_header[4..8], version);
        let Some(data_start) = at.checked_add(ID3V2_FRAME_HEADER_LEN) else {
            break;
        };
        let Some(next) = data_start.checked_add(size) else {
            break;
        };
        let Some(data) = body.get(data_start..next) else {
            break; // a size past the tag body: stop rather than read on
        };
        let unreadable = match version {
            ID3V2_VERSION_2_3 => ID3V2_3_FRAME_UNREADABLE,
            _ => ID3V2_4_FRAME_UNREADABLE,
        };
        if frame_header[9] & unreadable == 0 {
            if let Some(tag) = frame_to_tag(&frame_header[..4], data) {
                tags.push(tag);
            }
        }
        at = next;
    }
    tags
}

/// Byte length of the extended header at the start of an ID3v2 tag body, or
/// `None` when it does not fit. ID3v2.4 codes it as a syncsafe size that
/// includes the size field, ID3v2.3 as a plain size that excludes it.
fn extended_header_len(body: &[u8], version: u8) -> Option<usize> {
    let size = body.get(..4)?;
    let len = match version {
        ID3V2_VERSION_2_4 => syncsafe_u32(size)? as usize,
        _ => u32::from_be_bytes([size[0], size[1], size[2], size[3]]) as usize + 4,
    };
    (len <= body.len()).then_some(len)
}

/// The declared size of one frame's data. ID3v2.4 codes it syncsafe; ID3v2.3 as
/// a plain big-endian integer. A 2.4 size with a top bit set is not syncsafe at
/// all (some encoders write the 2.3 form under a 2.4 header), so it is read as
/// plain, which is what every reader does.
fn frame_size(bytes: &[u8], version: u8) -> usize {
    let plain = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    match version {
        ID3V2_VERSION_2_4 => syncsafe_u32(bytes).map_or(plain, |v| v as usize),
        _ => plain,
    }
}

/// Map one ID3v2 text frame to a [`Tag`], or `None` when the frame is not one
/// this reads or its text is empty. A frame with a typed [`Tag`] variant becomes
/// that variant; the rest keep their frame id as the key.
fn frame_to_tag(id: &[u8], data: &[u8]) -> Option<Tag> {
    let text = decode_text(data)?;
    if text.is_empty() {
        return None;
    }
    let tag = match id {
        b"TIT2" => Tag::Title(text),
        b"TPE1" => Tag::Artist(text),
        b"TALB" => Tag::Album(text),
        b"TSSE" => Tag::Encoder(text),
        b"TRCK" => track_number(&text).unwrap_or(Tag::Other {
            key: String::from_utf8_lossy(id).into_owned(),
            value: text,
        }),
        b"TYER" | b"TDRC" | b"TCON" | b"TPE2" | b"TCOM" => Tag::Other {
            key: String::from_utf8_lossy(id).into_owned(),
            value: text,
        },
        _ => return None,
    };
    Some(tag)
}

/// The leading integer of a `TRCK` value (`"3"` or `"3/12"`) as the typed track
/// number, or `None` when it does not start with digits.
fn track_number(text: &str) -> Option<Tag> {
    let digits: String = text.chars().take_while(char::is_ascii_digit).collect();
    let value = digits.parse().ok()?;
    Some(Tag::Number {
        key: String::from(Tag::TRACK_NUMBER),
        value,
    })
}

/// Decode an ID3v2 text frame's payload: an encoding byte then the text, in
/// ISO-8859-1, UTF-16 with a byte-order mark, big-endian UTF-16, or UTF-8. A
/// trailing terminator (and anything after it, which only multi-value frames
/// carry) is dropped. `None` for an empty payload or an unknown encoding.
fn decode_text(data: &[u8]) -> Option<String> {
    let (&encoding, text) = data.split_first()?;
    let decoded = match encoding {
        ENCODING_LATIN1 => text.iter().map(|&b| char::from(b)).collect(),
        ENCODING_UTF8 => String::from_utf8_lossy(text).into_owned(),
        ENCODING_UTF16_BOM | ENCODING_UTF16BE => decode_utf16(text, encoding),
        _ => return None,
    };
    Some(String::from(decoded.split('\0').next().unwrap_or("")))
}

/// Decode UTF-16 text: little-endian when a `FF FE` byte-order mark says so,
/// big-endian otherwise (the mark is absent from a `UTF16BE` frame). An unpaired
/// trailing byte and any unpaired surrogate are dropped.
fn decode_utf16(text: &[u8], encoding: u8) -> String {
    let (little_endian, body) = match text {
        [0xFF, 0xFE, rest @ ..] if encoding == ENCODING_UTF16_BOM => (true, rest),
        [0xFE, 0xFF, rest @ ..] if encoding == ENCODING_UTF16_BOM => (false, rest),
        _ => (false, text),
    };
    let units: Vec<u16> = body
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            if little_endian {
                u16::from_le_bytes(*pair)
            } else {
                u16::from_be_bytes(*pair)
            }
        })
        .collect();
    char::decode_utf16(units)
        .filter_map(Result::ok)
        .collect::<String>()
}

/// Parse the 128-byte ID3v1 block `buf` opens with, or `None` when it does not
/// open on one. The genre byte is not read: it indexes a list of names this does
/// not carry.
pub(crate) fn parse_id3v1(buf: &[u8]) -> Option<TagList> {
    let block = buf.get(..ID3V1_LEN)?;
    if block[..ID3V1_MAGIC.len()] != ID3V1_MAGIC {
        return None;
    }
    let mut rest = &block[ID3V1_MAGIC.len()..];
    let mut field = |len: usize| {
        let (text, tail) = rest.split_at(len);
        rest = tail;
        latin1_field(text)
    };
    let title = field(ID3V1_TEXT_LEN);
    let artist = field(ID3V1_TEXT_LEN);
    let album = field(ID3V1_TEXT_LEN);
    let year = field(ID3V1_YEAR_LEN);
    let comment = field(ID3V1_TEXT_LEN);
    let mut tags = TagList::new();
    push_if_set(&mut tags, title, Tag::Title);
    push_if_set(&mut tags, artist, Tag::Artist);
    push_if_set(&mut tags, album, Tag::Album);
    push_if_set(&mut tags, comment, Tag::Comment);
    if !year.is_empty() {
        tags.push(Tag::Other {
            key: String::from(ID3V1_YEAR_KEY),
            value: year,
        });
    }
    Some(tags)
}

/// The key an ID3v1 year is published under. ID3v2 keeps its frame id, and the
/// v1 block has no frame ids, so the recording year needs a name of its own.
const ID3V1_YEAR_KEY: &str = "date";

fn push_if_set(tags: &mut TagList, text: String, build: fn(String) -> Tag) {
    if !text.is_empty() {
        tags.push(build(text));
    }
}

/// One fixed-width ID3v1 field as text: ISO-8859-1, padded with NULs or spaces.
fn latin1_field(bytes: &[u8]) -> String {
    bytes
        .iter()
        .take_while(|&&b| b != 0)
        .map(|&b| char::from(b))
        .collect::<String>()
        .trim_end()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// An ID3v2 tag: header (`version`, `flags`) then `frames` verbatim, the
    /// body size coded syncsafe.
    fn id3v2(version: u8, flags: u8, frames: &[u8]) -> Vec<u8> {
        let size = frames.len() as u32;
        let mut tag = Vec::from(ID3V2_MAGIC);
        tag.extend_from_slice(&[version, 0, flags]);
        for shift in [21, 14, 7, 0] {
            tag.push(((size >> shift) & 0x7F) as u8);
        }
        tag.extend_from_slice(frames);
        tag
    }

    /// One frame: 4-byte id, size (syncsafe for 2.4, plain for 2.3), 2 flag
    /// bytes, then the payload.
    fn frame(version: u8, id: &[u8; 4], flags: u8, data: &[u8]) -> Vec<u8> {
        let size = data.len() as u32;
        let bits = if version == ID3V2_VERSION_2_4 {
            [21, 14, 7, 0]
        } else {
            [24, 16, 8, 0]
        };
        let mask = if version == ID3V2_VERSION_2_4 {
            0x7F
        } else {
            0xFF
        };
        let mut out = Vec::from(&id[..]);
        for shift in bits {
            out.push(((size >> shift) & mask) as u8);
        }
        out.extend_from_slice(&[0, flags]);
        out.extend_from_slice(data);
        out
    }

    fn latin1_frame(version: u8, id: &[u8; 4], text: &str) -> Vec<u8> {
        let mut data = vec![ENCODING_LATIN1];
        data.extend_from_slice(text.as_bytes());
        frame(version, id, 0, &data)
    }

    #[test]
    fn syncsafe_size_decodes_seven_bits_per_byte() {
        assert_eq!(syncsafe_u32(&[0, 0, 2, 1]), Some(257));
        assert_eq!(syncsafe_u32(&[0, 0, 0x7F, 0x7F]), Some(16_383));
        // A set top bit is not a syncsafe integer.
        assert_eq!(syncsafe_u32(&[0, 0, 0x80, 0]), None);
        assert_eq!(syncsafe_u32(&[0, 0, 0]), None);
    }

    #[test]
    fn tag_length_covers_header_body_and_footer() {
        let tag = id3v2(ID3V2_VERSION_2_3, 0, &[0u8; 300]);
        assert_eq!(id3v2_len(&tag), Some(ID3V2_HEADER_LEN + 300));
        let footed = id3v2(ID3V2_VERSION_2_4, ID3V2_FLAG_FOOTER, &[0u8; 40]);
        assert_eq!(
            id3v2_len(&footed),
            Some(ID3V2_HEADER_LEN + 40 + ID3V2_HEADER_LEN)
        );
        // Not a tag, and a header cut short.
        assert_eq!(id3v2_len(b"\xff\xfb\x90\x00"), None);
        assert_eq!(id3v2_len(&tag[..ID3V2_HEADER_LEN - 1]), None);
    }

    #[test]
    fn reads_id3v2_3_text_frames() {
        let mut frames = latin1_frame(ID3V2_VERSION_2_3, b"TIT2", "Sine");
        frames.extend(latin1_frame(ID3V2_VERSION_2_3, b"TPE1", "g2g"));
        frames.extend(latin1_frame(ID3V2_VERSION_2_3, b"TRCK", "3/12"));
        let tags = parse_id3v2(&id3v2(ID3V2_VERSION_2_3, 0, &frames));
        assert_eq!(
            tags.tags(),
            [
                Tag::Title("Sine".into()),
                Tag::Artist("g2g".into()),
                Tag::Number {
                    key: Tag::TRACK_NUMBER.into(),
                    value: 3
                },
            ]
        );
    }

    #[test]
    fn reads_id3v2_4_syncsafe_frame_sizes() {
        // A 200-byte payload: the syncsafe and plain codings differ above 127,
        // so a misread size would swallow the frame behind it.
        let long = "x".repeat(200);
        let mut frames = latin1_frame(ID3V2_VERSION_2_4, b"TALB", &long);
        frames.extend(latin1_frame(ID3V2_VERSION_2_4, b"TPE1", "g2g"));
        let tags = parse_id3v2(&id3v2(ID3V2_VERSION_2_4, 0, &frames));
        assert_eq!(
            tags.tags(),
            [Tag::Album(long.clone()), Tag::Artist("g2g".into())]
        );
    }

    #[test]
    fn reads_utf16_text_in_both_byte_orders() {
        let utf16 = |bom: [u8; 2], text: &str| {
            let mut data = vec![ENCODING_UTF16_BOM, bom[0], bom[1]];
            for unit in text.encode_utf16() {
                let bytes = if bom == [0xFF, 0xFE] {
                    unit.to_le_bytes()
                } else {
                    unit.to_be_bytes()
                };
                data.extend_from_slice(&bytes);
            }
            frame(ID3V2_VERSION_2_3, b"TIT2", 0, &data)
        };
        let little = parse_id3v2(&id3v2(
            ID3V2_VERSION_2_3,
            0,
            &utf16([0xFF, 0xFE], "Sinus \u{e4}"),
        ));
        assert_eq!(little.tags(), [Tag::Title("Sinus \u{e4}".into())]);
        let big = parse_id3v2(&id3v2(
            ID3V2_VERSION_2_3,
            0,
            &utf16([0xFE, 0xFF], "Sinus \u{e4}"),
        ));
        assert_eq!(big.tags(), [Tag::Title("Sinus \u{e4}".into())]);
        // No byte-order mark: UTF16BE.
        let mut be = vec![ENCODING_UTF16BE];
        be.extend_from_slice(&[0x00, 0x41, 0x00, 0x42]);
        let tags = parse_id3v2(&id3v2(
            ID3V2_VERSION_2_3,
            0,
            &frame(ID3V2_VERSION_2_3, b"TIT2", 0, &be),
        ));
        assert_eq!(tags.tags(), [Tag::Title("AB".into())]);
    }

    #[test]
    fn skips_compressed_and_encrypted_frames() {
        let mut data = vec![ENCODING_LATIN1];
        data.extend_from_slice(b"Sine");
        let mut frames = frame(ID3V2_VERSION_2_3, b"TIT2", ID3V2_3_FRAME_UNREADABLE, &data);
        frames.extend(latin1_frame(ID3V2_VERSION_2_3, b"TPE1", "g2g"));
        let tags = parse_id3v2(&id3v2(ID3V2_VERSION_2_3, 0, &frames));
        assert_eq!(tags.tags(), [Tag::Artist("g2g".into())]);
    }

    #[test]
    fn truncated_tag_yields_no_tags_and_no_panic() {
        let frames = latin1_frame(ID3V2_VERSION_2_3, b"TIT2", "Sine");
        let tag = id3v2(ID3V2_VERSION_2_3, 0, &frames);
        // Every prefix of a real tag: nothing may panic, and a frame whose data
        // has not all arrived is not reported.
        for cut in 0..tag.len() {
            let tags = parse_id3v2(&tag[..cut]);
            assert!(tags.is_empty(), "prefix of {cut} bytes reported {tags:?}");
        }
        assert_eq!(parse_id3v2(&tag).len(), 1);
        // A frame size far past the body ends the walk.
        let mut lying = tag.clone();
        lying[ID3V2_HEADER_LEN + 4] = 0x7F;
        assert!(parse_id3v2(&lying).is_empty());
    }

    #[test]
    fn reads_the_id3v1_trailer() {
        let mut block = Vec::from(ID3V1_MAGIC);
        block.extend_from_slice(b"Sine");
        block.resize(ID3V1_MAGIC.len() + ID3V1_TEXT_LEN, 0);
        block.extend_from_slice(b"g2g");
        block.resize(ID3V1_MAGIC.len() + 3 * ID3V1_TEXT_LEN, 0);
        block.extend_from_slice(b"2026");
        block.resize(ID3V1_LEN, 0);
        let tags = parse_id3v1(&block).expect("a TAG block parses");
        assert_eq!(
            tags.tags(),
            [
                Tag::Title("Sine".into()),
                Tag::Artist("g2g".into()),
                Tag::Other {
                    key: ID3V1_YEAR_KEY.into(),
                    value: "2026".into()
                },
            ]
        );
        assert!(parse_id3v1(&block[..ID3V1_LEN - 1]).is_none());
        assert!(parse_id3v1(&[0u8; ID3V1_LEN]).is_none());
    }
}
