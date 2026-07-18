//! Stream metadata tags (the GStreamer `GstTagList` analog).
//!
//! A demuxer surfaces a stream's descriptive metadata (title, artist, encoder,
//! ...) as a [`TagList`], delivered to the application out of band on the bus
//! ([`BusMessage::Tag`](crate::bus::BusMessage::Tag)). Common keys are typed;
//! anything else keeps its container-native name in [`Tag::Other`], so a tag a
//! given container defines but this enum doesn't still round-trips.
//!
//! `no_std + alloc`: the type is in the baseline so any element can build one,
//! even though today only the bus (a `runtime` feature) carries it.

use alloc::string::String;
use alloc::vec::Vec;

/// One stream-metadata tag. The typed variants are the cross-container common
/// keys (VorbisComment / Matroska / FLV `onMetaData` all define them under the
/// same names); [`Tag::Other`] carries any other key verbatim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tag {
    Title(String),
    Artist(String),
    Album(String),
    /// The encoding application or library.
    Encoder(String),
    Language(String),
    Comment(String),
    /// A tag whose key is not one of the typed variants.
    Other {
        key: String,
        value: String,
    },
}

impl Tag {
    /// Map a `key`/`value` pair to a typed tag, or [`Tag::Other`] when the key
    /// is unrecognized. The key match is ASCII case-insensitive, since the
    /// container metadata formats that feed this treat keys that way.
    pub fn from_key_value(key: &str, value: &str) -> Tag {
        let v = String::from(value);
        if key.eq_ignore_ascii_case("title") {
            Tag::Title(v)
        } else if key.eq_ignore_ascii_case("artist") {
            Tag::Artist(v)
        } else if key.eq_ignore_ascii_case("album") {
            Tag::Album(v)
        } else if key.eq_ignore_ascii_case("encoder") {
            Tag::Encoder(v)
        } else if key.eq_ignore_ascii_case("language") {
            Tag::Language(v)
        } else if key.eq_ignore_ascii_case("comment") || key.eq_ignore_ascii_case("description") {
            Tag::Comment(v)
        } else {
            Tag::Other {
                key: String::from(key),
                value: v,
            }
        }
    }
}

/// An ordered, deduplication-free list of [`Tag`]s for one stream.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TagList {
    tags: Vec<Tag>,
}

impl TagList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, tag: Tag) {
        self.tags.push(tag);
    }

    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tags.len()
    }

    /// The tags in insertion order.
    pub fn tags(&self) -> &[Tag] {
        &self.tags
    }
}

impl FromIterator<Tag> for TagList {
    fn from_iter<I: IntoIterator<Item = Tag>>(iter: I) -> Self {
        Self {
            tags: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_known_keys_case_insensitively() {
        assert_eq!(
            Tag::from_key_value("TITLE", "Song"),
            Tag::Title("Song".into())
        );
        assert_eq!(
            Tag::from_key_value("Artist", "Band"),
            Tag::Artist("Band".into())
        );
        assert_eq!(
            Tag::from_key_value("encoder", "libopus"),
            Tag::Encoder("libopus".into())
        );
        assert_eq!(
            Tag::from_key_value("DESCRIPTION", "hi"),
            Tag::Comment("hi".into())
        );
    }

    #[test]
    fn unknown_key_falls_back_to_other() {
        assert_eq!(
            Tag::from_key_value("REPLAYGAIN_TRACK_GAIN", "-3.2 dB"),
            Tag::Other {
                key: "REPLAYGAIN_TRACK_GAIN".into(),
                value: "-3.2 dB".into()
            }
        );
    }

    #[test]
    fn taglist_collects_and_reports() {
        let list: TagList = [Tag::Title("T".into()), Tag::Artist("A".into())]
            .into_iter()
            .collect();
        assert_eq!(list.len(), 2);
        assert!(!list.is_empty());
        assert_eq!(list.tags()[0], Tag::Title("T".into()));
        assert!(TagList::new().is_empty());
    }
}
