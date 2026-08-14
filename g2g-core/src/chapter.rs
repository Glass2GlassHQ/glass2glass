//! Container chapters, the table of contents a file declares (the GStreamer
//! `GstToc` analog).
//!
//! A demuxer surfaces them out of band on the bus
//! ([`BusMessage::Chapters`](crate::bus::BusMessage::Chapters)), the same way it
//! surfaces [`TagList`](crate::tag::TagList) metadata: the application gets the
//! seek points and their titles without intercepting the data path. A muxer
//! takes the same list back and writes the container's chapter element.
//!
//! `no_std + alloc`: the type is in the baseline so any element can build one,
//! even though today only the bus (a `runtime` feature) carries it.
//!
//! # Timing
//!
//! [`start_ns`](Chapter::start_ns) and [`end_ns`](Chapter::end_ns) are stream
//! time in nanoseconds, whatever units the container stored them in (Matroska
//! `ChapterTimeStart` is already nanoseconds, a Nero `chpl` counts 100 ns ticks,
//! a QuickTime chapter track counts its own timescale). `end_ns` is `None` when
//! the container carries no end for the entry: a `chpl` chapter runs until the
//! next one starts, which only the reader of the whole list can work out.

use alloc::string::String;
use alloc::vec::Vec;

/// One chapter: a named point on the stream timeline, optionally bounded and
/// optionally holding nested chapters (a Matroska `ChapterAtom` may contain
/// further atoms; the flat containers give every chapter an empty
/// [`sub_chapters`](Chapter::sub_chapters)).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Chapter {
    /// Where the chapter starts, in stream-time nanoseconds.
    pub start_ns: u64,
    /// Where it ends, in stream-time nanoseconds, when the container says.
    pub end_ns: Option<u64>,
    /// The chapter's display title, empty when it declares none.
    pub title: String,
    /// The title's language as the container spelled it (a Matroska
    /// `ChapLanguage`, an ISO 639-2 code like `eng`). `None` for a container
    /// with no per-chapter language (Nero `chpl`, a QuickTime chapter track).
    pub language: Option<String>,
    /// Chapters nested inside this one, in file order.
    pub sub_chapters: Vec<Chapter>,
}

impl Chapter {
    /// A chapter starting at `start_ns` titled `title`, with no end, language, or
    /// nesting: the shape the flat containers produce.
    pub fn new(start_ns: u64, title: impl Into<String>) -> Self {
        Self {
            start_ns,
            title: title.into(),
            ..Self::default()
        }
    }

    /// The same chapter bounded at `end_ns`.
    pub fn with_end_ns(mut self, end_ns: u64) -> Self {
        self.end_ns = Some(end_ns);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_set_start_title_and_end() {
        let c = Chapter::new(2_000_000_000, "Middle Part").with_end_ns(5_000_000_000);
        assert_eq!(c.start_ns, 2_000_000_000);
        assert_eq!(c.end_ns, Some(5_000_000_000));
        assert_eq!(c.title, "Middle Part");
        assert_eq!(c.language, None);
        assert!(c.sub_chapters.is_empty());
    }
}
