//! Stream discovery: the available elementary streams of a container (the
//! GStreamer `GstStreamCollection` analog, the data model playbin is built on).
//!
//! A demuxer parses a container's track list and announces *every* elementary
//! stream it found, out of band on the bus
//! ([`BusMessage::StreamCollection`](crate::bus::BusMessage::StreamCollection)),
//! independent of which one(s) it actually forwards. The application reads the
//! collection to learn what audio / video / text streams exist (their type and
//! [`Caps`]) so it can later select among them. This is the discovery half of
//! the playbin model; app-driven selection is a follow-up.
//!
//! `no_std + alloc`: the types are in the baseline so any demuxer can build one,
//! even though today only the bus (a `runtime` feature) carries it. Mirrors the
//! sibling [`TagList`](crate::tag::TagList).

use alloc::string::String;
use alloc::vec::Vec;

use crate::caps::Caps;

/// The kind of media an elementary stream carries (the `GstStreamType` analog).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamType {
    Video,
    Audio,
    /// Subtitles / captions.
    Text,
    /// A stream whose kind the demuxer could not classify.
    Unknown,
}

/// One elementary stream of a container: a stable id, its media kind, and the
/// [`Caps`] it carries. The id is the cross-run-stable stream identifier (the
/// GStreamer stream-id analog), e.g. `"matroska-track-1"`, so an application can
/// name a stream to select it. Per-stream tags / flags are a deliberate future
/// extension (added when a consumer needs them; the struct stays additive).
#[derive(Clone, Debug, PartialEq)]
pub struct Stream {
    pub id: String,
    pub stream_type: StreamType,
    pub caps: Caps,
}

impl Stream {
    pub fn new(id: impl Into<String>, stream_type: StreamType, caps: Caps) -> Self {
        Self { id: id.into(), stream_type, caps }
    }
}

/// The set of elementary streams a demuxer found in one container (the
/// `GstStreamCollection` analog). Carries a collection id (the upstream / demuxer
/// identity) so an application can tell one collection from another when more
/// than one source feeds a pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamCollection {
    pub id: String,
    pub streams: Vec<Stream>,
}

impl StreamCollection {
    pub fn new(id: impl Into<String>, streams: Vec<Stream>) -> Self {
        Self { id: id.into(), streams }
    }

    /// All streams, in the demuxer's declared track order.
    pub fn streams(&self) -> &[Stream] {
        &self.streams
    }

    /// The streams of a given kind (e.g. every audio track), in track order.
    pub fn streams_of_type(&self, stream_type: StreamType) -> impl Iterator<Item = &Stream> {
        self.streams.iter().filter(move |s| s.stream_type == stream_type)
    }

    /// The stream with this id, if present.
    pub fn get(&self, stream_id: &str) -> Option<&Stream> {
        self.streams.iter().find(|s| s.id == stream_id)
    }

    pub fn len(&self) -> usize {
        self.streams.len()
    }

    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{AudioFormat, Dim, Rate, VideoCodec};

    fn video() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Any,
        }
    }
    fn audio() -> Caps {
        Caps::Audio { format: AudioFormat::Opus, channels: 2, sample_rate: 48_000 }
    }

    fn collection() -> StreamCollection {
        StreamCollection::new(
            "matroska-0",
            alloc::vec![
                Stream::new("matroska-track-1", StreamType::Video, video()),
                Stream::new("matroska-track-2", StreamType::Audio, audio()),
            ],
        )
    }

    #[test]
    fn streams_of_type_filters_by_kind() {
        let c = collection();
        let video: Vec<_> = c.streams_of_type(StreamType::Video).collect();
        assert_eq!(video.len(), 1);
        assert_eq!(video[0].id, "matroska-track-1");
        assert_eq!(c.streams_of_type(StreamType::Audio).count(), 1);
        assert_eq!(c.streams_of_type(StreamType::Text).count(), 0);
    }

    #[test]
    fn get_finds_by_id() {
        let c = collection();
        assert_eq!(c.get("matroska-track-2").unwrap().stream_type, StreamType::Audio);
        assert!(c.get("nonexistent").is_none());
    }

    #[test]
    fn len_and_empty() {
        assert_eq!(collection().len(), 2);
        assert!(!collection().is_empty());
        let empty = StreamCollection::new("x", Vec::new());
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }
}
