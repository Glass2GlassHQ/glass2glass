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
//!
//! # Global and per-stream tags
//!
//! A multi-stream container tags the file as a whole *and* each of its streams,
//! and the two stay distinct. A demuxer posts the container's own tags once as
//! [`BusMessage::Tag`](crate::bus::BusMessage::Tag) and each stream's as a
//! [`BusMessage::StreamTag`](crate::bus::BusMessage::StreamTag) carrying that
//! stream's id, never folding one into the other.
//!
//! The conflict rule: when the same key is set both container-globally and on a
//! stream, **the stream's tag wins on that stream's pad**, while the global one
//! still stands for every stream that does not set the key. [`resolve_tags`]
//! applies it for a consumer that wants one list per stream.
//!
//! A muxer runs the inverse split ([`split_tags`]): a tag every input carries
//! identically is written once at the container level, a tag already set
//! globally is not repeated on a stream, and everything else is written in that
//! stream's own per-track container (a Matroska `Targets`-scoped `Tag`, an MP4
//! `trak/udta/meta/ilst`).

use alloc::borrow::Cow;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// One stream-metadata tag. The typed variants are the cross-container common
/// keys (VorbisComment / Matroska / FLV `onMetaData` all define them under the
/// same names); [`Tag::Number`] carries an integer-valued key (an MP4 `trkn` /
/// `disk` / `tmpo` / `cpil` atom), [`Tag::Freeform`] a key namespaced by a
/// reverse-DNS owner (an MP4 `----` atom), and [`Tag::Other`] any other key
/// verbatim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tag {
    Title(String),
    Artist(String),
    Album(String),
    /// The encoding application or library.
    Encoder(String),
    Language(String),
    Comment(String),
    /// An integer-valued tag. The cross-container keys are the associated
    /// constants ([`Tag::TRACK_NUMBER`], ...); anything else keeps its
    /// container-native key.
    Number {
        key: String,
        value: u64,
    },
    /// A tag whose key is scoped by a reverse-DNS namespace (`com.apple.iTunes`
    /// and friends), the MP4 `----` atom's `mean` / `name` pair. A container
    /// with no namespace concept flattens it to the `namespace:key` string
    /// [`Tag::key`] reports.
    Freeform {
        namespace: String,
        key: String,
        value: String,
    },
    /// A tag whose key is not one of the typed variants.
    Other {
        key: String,
        value: String,
    },
}

impl Tag {
    /// Position of this track within its album (MP4 `trkn`).
    pub const TRACK_NUMBER: &'static str = "track-number";
    /// How many tracks the album holds (MP4 `trkn`'s second half).
    pub const TRACK_COUNT: &'static str = "track-count";
    /// Position of this disc within its set (MP4 `disk`).
    pub const DISC_NUMBER: &'static str = "album-disc-number";
    /// How many discs the set holds (MP4 `disk`'s second half).
    pub const DISC_COUNT: &'static str = "album-disc-count";
    /// Tempo in beats per minute (MP4 `tmpo`).
    pub const BEATS_PER_MINUTE: &'static str = "beats-per-minute";
    /// Part of a compilation, `0` or `1` (MP4 `cpil`).
    pub const COMPILATION: &'static str = "compilation";

    /// The tag's key: the conventional lowercase name of a typed variant, the
    /// stored key of a [`Number`](Tag::Number) / [`Other`](Tag::Other), or the
    /// `namespace:key` of a [`Freeform`](Tag::Freeform). Two tags with the same
    /// key are the same metadata, which is what the global / per-stream conflict
    /// rule turns on.
    pub fn key(&self) -> Cow<'_, str> {
        match self {
            Tag::Title(_) => Cow::Borrowed("title"),
            Tag::Artist(_) => Cow::Borrowed("artist"),
            Tag::Album(_) => Cow::Borrowed("album"),
            Tag::Encoder(_) => Cow::Borrowed("encoder"),
            Tag::Language(_) => Cow::Borrowed("language"),
            Tag::Comment(_) => Cow::Borrowed("comment"),
            Tag::Number { key, .. } | Tag::Other { key, .. } => Cow::Borrowed(key),
            Tag::Freeform { namespace, key, .. } => Cow::Owned(format!("{namespace}:{key}")),
        }
    }

    /// The tag's value as text: a [`Number`](Tag::Number) renders in decimal, so
    /// a container with only string-valued metadata (Matroska `SimpleTag`, FLV
    /// `onMetaData`) can carry every variant.
    pub fn value_string(&self) -> Cow<'_, str> {
        match self {
            Tag::Title(v)
            | Tag::Artist(v)
            | Tag::Album(v)
            | Tag::Encoder(v)
            | Tag::Language(v)
            | Tag::Comment(v)
            | Tag::Freeform { value: v, .. }
            | Tag::Other { value: v, .. } => Cow::Borrowed(v),
            Tag::Number { value, .. } => Cow::Owned(format!("{value}")),
        }
    }

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

/// The tags that apply to one stream: the container's `global` tags less any key
/// the stream sets itself, then the stream's own. The demux side of the merge
/// policy in this module's docs, for a consumer that wants the effective list
/// for one pad (a demuxer posts the two lists separately and never merges them).
pub fn resolve_tags(global: &TagList, stream: &TagList) -> TagList {
    let mut out = TagList::new();
    for tag in global.tags() {
        if !stream.tags().iter().any(|s| s.key() == tag.key()) {
            out.push(tag.clone());
        }
    }
    for tag in stream.tags() {
        out.push(tag.clone());
    }
    out
}

/// Split a muxer's tags into what the container-level slot carries and what each
/// stream's own slot carries: the mux side of the merge policy in this module's
/// docs. A tag every stream repeats identically joins `global` (written once);
/// a tag already in `global` drops out of the per-stream lists; anything left
/// stays on its stream, including a tag whose key `global` also sets (the stream
/// wins on its own pad).
pub fn split_tags(global: &TagList, per_stream: &[TagList]) -> (TagList, Vec<TagList>) {
    let mut out_global = global.clone();
    if per_stream.len() > 1 {
        for tag in per_stream[0].tags() {
            let shared = per_stream[1..].iter().all(|s| s.tags().contains(tag));
            if shared && !out_global.tags().contains(tag) {
                out_global.push(tag.clone());
            }
        }
    }
    let out_streams = per_stream
        .iter()
        .map(|s| {
            s.tags()
                .iter()
                .filter(|t| !out_global.tags().contains(t))
                .cloned()
                .collect()
        })
        .collect();
    (out_global, out_streams)
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
    fn number_and_freeform_report_key_and_value() {
        let n = Tag::Number {
            key: Tag::TRACK_NUMBER.into(),
            value: 3,
        };
        assert_eq!(n.key(), "track-number");
        assert_eq!(n.value_string(), "3");
        let f = Tag::Freeform {
            namespace: "com.apple.iTunes".into(),
            key: "MOOD".into(),
            value: "calm".into(),
        };
        assert_eq!(f.key(), "com.apple.iTunes:MOOD");
        assert_eq!(f.value_string(), "calm");
        assert_eq!(Tag::Title("T".into()).key(), "title");
    }

    /// The conflict rule: the stream's tag wins for a key both scopes set, and a
    /// global tag the stream leaves alone still applies.
    #[test]
    fn resolve_lets_the_stream_win_on_a_shared_key() {
        let global: TagList = [Tag::Title("File".into()), Tag::Album("Set".into())]
            .into_iter()
            .collect();
        let stream: TagList = [Tag::Title("Track".into())].into_iter().collect();
        let merged = resolve_tags(&global, &stream);
        assert_eq!(
            merged.tags(),
            &[Tag::Album("Set".into()), Tag::Title("Track".into())]
        );
    }

    #[test]
    fn split_hoists_shared_tags_and_keeps_stream_only_ones() {
        let global: TagList = [Tag::Album("Set".into())].into_iter().collect();
        let a: TagList = [
            Tag::Encoder("g2g".into()),
            Tag::Album("Set".into()),
            Tag::Title("Video".into()),
        ]
        .into_iter()
        .collect();
        let b: TagList = [Tag::Encoder("g2g".into()), Tag::Title("Audio".into())]
            .into_iter()
            .collect();
        let (out_global, streams) = split_tags(&global, &[a, b]);
        assert_eq!(
            out_global.tags(),
            &[Tag::Album("Set".into()), Tag::Encoder("g2g".into())],
            "the tag both streams repeat moves up, once"
        );
        assert_eq!(
            streams[0].tags(),
            &[Tag::Title("Video".into())],
            "the already-global album drops out, the stream-only title stays"
        );
        assert_eq!(streams[1].tags(), &[Tag::Title("Audio".into())]);
    }

    /// A per-stream tag whose key is also set globally stays on its stream: it
    /// is the override the conflict rule resolves.
    #[test]
    fn split_keeps_a_stream_override_of_a_global_key() {
        let global: TagList = [Tag::Title("File".into())].into_iter().collect();
        let a: TagList = [Tag::Title("Video".into())].into_iter().collect();
        let (out_global, streams) = split_tags(&global, &[a]);
        assert_eq!(out_global.tags(), &[Tag::Title("File".into())]);
        assert_eq!(streams[0].tags(), &[Tag::Title("Video".into())]);
        assert_eq!(
            resolve_tags(&out_global, &streams[0]).tags(),
            &[Tag::Title("Video".into())]
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
