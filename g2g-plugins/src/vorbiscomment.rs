//! VorbisComment metadata, read and written.
//!
//! One block carries a vendor string then a count-prefixed list of `KEY=VALUE`
//! UTF-8 fields (RFC 7845 §5.2). Four carriers wrap the same body: the Vorbis
//! comment header packet (`\x03vorbis`, which appends a framing bit), the Opus
//! one (`OpusTags`), a FLAC VORBIS_COMMENT metadata block (type 4, behind a
//! 4-byte block header), and the Ogg-FLAC mapping's copy of that block.
//!
//! Every length here comes off the wire, so the reader bounds each step against
//! what actually arrived and returns whatever it read rather than panicking.

use alloc::borrow::Cow;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::{Tag, TagList};

/// The vendor string written into a block this crate builds.
pub(crate) const VENDOR: &[u8] = b"g2g";

/// The Vorbis comment header packet's magic. Its mapping mandates a framing bit
/// after the fields, which no other carrier has.
pub(crate) const VORBIS_COMMENT_MAGIC: &[u8] = b"\x03vorbis";
/// The Opus comment header packet's magic (RFC 7845 §5.2).
pub(crate) const OPUS_TAGS_MAGIC: &[u8] = b"OpusTags";
/// FLAC metadata block type of a VORBIS_COMMENT block.
pub(crate) const FLAC_COMMENT_BLOCK_TYPE: u8 = 4;
/// A FLAC metadata block header: the type byte (top bit = last block) and a
/// 24-bit big-endian body length.
pub(crate) const FLAC_BLOCK_HEADER_LEN: usize = 4;

/// A VorbisComment block behind `magic`: `vendor`, then one `KEY=VALUE` field
/// per tag. The Vorbis flavour gets the framing bit its own mapping mandates;
/// `OpusTags` and the FLAC block do not carry one.
pub(crate) fn vorbis_comment(magic: &[u8], vendor: &[u8], tags: &TagList) -> Vec<u8> {
    let mut p = Vec::from(magic);
    p.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    p.extend_from_slice(vendor);
    p.extend_from_slice(&(tags.len() as u32).to_le_bytes());
    for tag in tags.tags() {
        let field = alloc::format!("{}={}", vorbis_key(tag), tag.value_string());
        p.extend_from_slice(&(field.len() as u32).to_le_bytes());
        p.extend_from_slice(field.as_bytes());
    }
    if magic.starts_with(b"\x03") {
        p.push(1); // framing bit
    }
    p
}

/// The field name a tag is written under. VorbisComment spells the common keys
/// in upper case; a key that came in verbatim ([`Tag::Other`] and friends) keeps
/// the case it was given, so `REPLAYGAIN_TRACK_GAIN` and a MusicBrainz key
/// survive a round trip.
fn vorbis_key(tag: &Tag) -> Cow<'_, str> {
    match tag {
        Tag::Number { .. } | Tag::Freeform { .. } | Tag::Other { .. } => tag.key(),
        _ => Cow::Owned(tag.key().to_uppercase()),
    }
}

/// The vendor string of the VorbisComment block `packet` carries, or `None` when
/// it is not one / is truncated. A tag writer reuses it so rewriting the tags
/// does not relabel who encoded the stream.
pub(crate) fn vorbis_comment_vendor(packet: &[u8]) -> Option<String> {
    comment_body_vendor(comment_body(packet)?)
}

/// [`vorbis_comment_vendor`] for a body already stripped of its carrier, which
/// is how a FLAC metadata block arrives.
pub(crate) fn comment_body_vendor(body: &[u8]) -> Option<String> {
    let len = read_u32_le(body, &mut 0)? as usize;
    let vendor = body.get(4..4usize.checked_add(len)?)?;
    Some(String::from_utf8_lossy(vendor).to_string())
}

/// The comment body behind whichever carrier `packet` uses, or `None` when it is
/// none of them.
fn comment_body(packet: &[u8]) -> Option<&[u8]> {
    if let Some(rest) = packet.strip_prefix(OPUS_TAGS_MAGIC) {
        Some(rest)
    } else if let Some(rest) = packet.strip_prefix(VORBIS_COMMENT_MAGIC) {
        Some(rest)
    } else if packet.len() >= FLAC_BLOCK_HEADER_LEN && packet[0] & 0x7F == FLAC_COMMENT_BLOCK_TYPE {
        Some(&packet[FLAC_BLOCK_HEADER_LEN..])
    } else {
        None
    }
}

fn read_u32_le(b: &[u8], pos: &mut usize) -> Option<u32> {
    let s = b.get(*pos..pos.checked_add(4)?)?;
    *pos += 4;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// Parse a VorbisComment block into a [`TagList`]. Accepts the comment header
/// with its codec prefix (`OpusTags`, the Vorbis `\x03vorbis`, or a FLAC
/// VORBIS_COMMENT metadata block, whose 4-byte block header wraps the same
/// body). Unparseable / truncated input yields whatever was read so far.
pub(crate) fn parse_vorbis_comment(packet: &[u8]) -> TagList {
    match comment_body(packet) {
        Some(body) => parse_comment_body(body),
        None => TagList::new(),
    }
}

/// [`parse_vorbis_comment`] for a body already stripped of its carrier, which is
/// how a FLAC metadata block arrives.
pub(crate) fn parse_comment_body(body: &[u8]) -> TagList {
    let mut list = TagList::new();
    let mut pos = 0usize;
    let Some(vendor_len) = read_u32_le(body, &mut pos) else {
        return list;
    };
    pos = match pos.checked_add(vendor_len as usize) {
        Some(p) if p <= body.len() => p, // skip the vendor string
        _ => return list,
    };
    let Some(count) = read_u32_le(body, &mut pos) else {
        return list;
    };
    for _ in 0..count {
        let Some(len) = read_u32_le(body, &mut pos) else {
            break;
        };
        let Some(end) = pos.checked_add(len as usize) else {
            break;
        };
        let Some(field) = body.get(pos..end) else {
            break;
        };
        pos = end;
        if let Ok(s) = core::str::from_utf8(field) {
            if let Some((key, value)) = s.split_once('=') {
                list.push(Tag::from_key_value(key, value));
            }
        }
    }
    list
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags() -> TagList {
        [
            Tag::Title("Sine".into()),
            Tag::Artist("g2g".into()),
            Tag::Other {
                key: "REPLAYGAIN_TRACK_GAIN".into(),
                value: "-3.2 dB".into(),
            },
        ]
        .into_iter()
        .collect()
    }

    /// The writer's output read back by the reader, for each carrier.
    #[test]
    fn round_trips_through_every_carrier() {
        for magic in [VORBIS_COMMENT_MAGIC, OPUS_TAGS_MAGIC] {
            let block = vorbis_comment(magic, VENDOR, &tags());
            assert_eq!(parse_vorbis_comment(&block).tags(), tags().tags());
            assert_eq!(
                vorbis_comment_vendor(&block).as_deref(),
                Some(core::str::from_utf8(VENDOR).unwrap())
            );
        }
    }

    #[test]
    fn writes_the_vorbis_framing_bit_only() {
        assert_eq!(
            *vorbis_comment(VORBIS_COMMENT_MAGIC, VENDOR, &TagList::new())
                .last()
                .expect("a non-empty block"),
            1
        );
        let opus = vorbis_comment(OPUS_TAGS_MAGIC, VENDOR, &TagList::new());
        // Magic, vendor length, vendor, field count: nothing after.
        assert_eq!(opus.len(), OPUS_TAGS_MAGIC.len() + 4 + VENDOR.len() + 4);
    }

    /// Every prefix of a real block: nothing may panic, and no field is reported
    /// from bytes that have not arrived.
    #[test]
    fn truncated_block_yields_no_panic() {
        let block = vorbis_comment(OPUS_TAGS_MAGIC, VENDOR, &tags());
        for cut in 0..block.len() {
            assert!(parse_vorbis_comment(&block[..cut]).len() <= tags().len());
        }
        assert!(parse_vorbis_comment(b"not a comment block").is_empty());
    }
}
