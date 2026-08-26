//! ID3 tag stripper element (`id3demux`): the tagged byte stream in, the same
//! bytes without the tags out.
//!
//! An ID3v2 tag ahead of a stream and an ID3v1 block behind it are metadata
//! glued onto whatever the file carries, and a parser fed them would have to
//! resynchronize past them. [`crate::mpegaudioparse`] skips them itself (an
//! `.mp3` reaches it directly), so this element is for putting the tags in front
//! of a parser that does not: the bytes it forwards are the payload alone.
//!
//! The caps do not change: the stream behind the tags is the stream that was
//! declared. The tags reach the application on the bus, once.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::short_type_name;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, BusHandle, BusMessage, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PipelinePacket, TagList,
};

use crate::id3::{id3v2_len, parse_id3v1, parse_id3v2, ID3V1_LEN, ID3V2_HEADER_LEN};

/// The largest ID3v2 tag whose text frames are read, as in
/// [`crate::mpegaudioparse`]: a tag past this is carrying artwork.
const MAX_ID3V2_TAG_PARSED: usize = 1 << 20;

/// # Example
///
/// ```no_run
/// use g2g_plugins::id3demux::Id3Demux;
///
/// let element = Id3Demux::new();
/// assert!(element.tags().is_empty());
/// ```
#[derive(Debug, Default)]
pub struct Id3Demux {
    configured: bool,
    bus: Option<BusHandle>,
    /// Bytes not yet forwarded: the head until the tag is decided, then the
    /// trailing [`ID3V1_LEN`] bytes that may be the ID3v1 block.
    buf: Vec<u8>,
    head_examined: bool,
    /// Bytes of ID3v2 tag still to drop from the front of the stream.
    id3v2_skip: usize,
    tags: TagList,
    tags_posted: bool,
    sequence: u64,
}

impl Id3Demux {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach the pipeline bus so the stream's ID3 tags reach the application as
    /// a [`BusMessage::Tag`].
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// The ID3 tags read from the stream, empty until one is parsed.
    pub fn tags(&self) -> &TagList {
        &self.tags
    }

    fn post_tags(&mut self) {
        if self.tags_posted || self.tags.is_empty() {
            return;
        }
        self.tags_posted = true;
        if let Some(bus) = &self.bus {
            bus.try_post(BusMessage::Tag {
                tags: self.tags.clone(),
                program: None,
            });
        }
    }

    /// Skip the ID3v2 tag at the head of the stream, reading its text frames on
    /// the way past. Returns whether the payload has been reached.
    fn skip_id3v2(&mut self, eos: bool) -> bool {
        if !self.head_examined {
            if self.buf.len() < ID3V2_HEADER_LEN && !eos {
                return false;
            }
            self.head_examined = true;
            self.id3v2_skip = id3v2_len(&self.buf).unwrap_or(0);
        }
        if self.id3v2_skip == 0 {
            return true;
        }
        let readable = self.id3v2_skip <= MAX_ID3V2_TAG_PARSED;
        if readable && self.buf.len() < self.id3v2_skip {
            return false; // the tags are read from the whole tag: wait for it
        }
        if readable {
            self.tags = parse_id3v2(&self.buf[..self.id3v2_skip]);
        }
        let drop = self.id3v2_skip.min(self.buf.len());
        self.buf.drain(..drop);
        self.id3v2_skip -= drop;
        self.id3v2_skip == 0
    }

    /// Forward everything settled: the payload past the ID3v2 tag, less the
    /// trailing bytes that may still turn out to be the ID3v1 block.
    async fn drain(&mut self, eos: bool, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if !self.skip_id3v2(eos) {
            return Ok(());
        }
        self.post_tags();
        let hold = if eos { 0 } else { ID3V1_LEN };
        let Some(len) = self.buf.len().checked_sub(hold) else {
            return Ok(());
        };
        if len == 0 {
            return Ok(());
        }
        let data: Vec<u8> = self.buf.drain(..len).collect();
        // Unstamped: a buffer here can carry bytes from two inputs, so neither
        // input's time describes it. The parser behind this stamps from the
        // frame headers.
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            Default::default(),
            self.sequence,
        );
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    /// Drop the ID3v1 block at the end of the stream, reading its fields.
    fn strip_id3v1(&mut self) {
        let Some(start) = self.buf.len().checked_sub(ID3V1_LEN) else {
            return;
        };
        let Some(tags) = parse_id3v1(&self.buf[start..]) else {
            return;
        };
        if self.tags.is_empty() {
            self.tags = tags;
        }
        self.buf.truncate(start);
    }
}

impl AsyncElement for Id3Demux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    /// The tags are not part of the media type, so whatever was declared for the
    /// tagged stream is what the stripped stream is.
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ID3 tag demuxer",
            "Codec/Demuxer/Metadata",
            "Strips ID3v1 / ID3v2 tags from a stream and posts them on the bus",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    self.buf.extend_from_slice(slice);
                    self.drain(false, out).await?;
                }
                // Only the end of the stream can tell a trailing `TAG` block
                // from payload.
                PipelinePacket::Eos => {
                    self.strip_id3v1();
                    self.drain(true, out).await?;
                    self.post_tags();
                }
                PipelinePacket::Flush => {
                    self.buf.clear();
                    out.push(PipelinePacket::Flush).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}
