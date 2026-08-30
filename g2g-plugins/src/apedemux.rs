//! APEv2 tag stripper (`apedemux`): the tagged byte stream in, the same bytes
//! without the tag out. The read side of [`crate::apev2mux::ApeV2Mux`].
//!
//! An APEv2 tag may sit at the start or end of the file. A trailing tag goes
//! ahead of an ID3v1 block when there is one. This element holds the tail of the
//! stream back until EOS, reads the tag's text items, posts them on the bus, and
//! forwards the payload (plus any ID3v1 block) without the APEv2 tag.
//!
//! The caps do not change: the stream behind the tag is the stream that was
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

use crate::apev2mux::{apev2_start_len, parse_apev2_tail, APE_FOOTER_LEN, APE_PREAMBLE, TAIL_HELD};
use crate::id3::{parse_id3v1, ID3V1_LEN};

/// # Example
///
/// ```no_run
/// use g2g_plugins::apedemux::ApeDemux;
///
/// let element = ApeDemux::new();
/// assert!(element.tags().is_empty());
/// ```
#[derive(Debug, Default)]
pub struct ApeDemux {
    configured: bool,
    bus: Option<BusHandle>,
    buf: Vec<u8>,
    tags: TagList,
    start_checked: bool,
    sequence: u64,
}

impl ApeDemux {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach the pipeline bus so the stream's APEv2 tags reach the application
    /// as a [`BusMessage::Tag`].
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// The APEv2 tags read from the stream, empty until EOS parses one.
    pub fn tags(&self) -> &TagList {
        &self.tags
    }

    async fn emit(&mut self, data: Vec<u8>, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if data.is_empty() {
            return Ok(());
        }
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            Default::default(),
            self.sequence,
        );
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    /// Forward everything but the held-back tail.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if !self.start_checked {
            if self.buf.len() < APE_PREAMBLE.len() {
                return Ok(());
            }
            if !self.buf.starts_with(APE_PREAMBLE) {
                self.start_checked = true;
            } else {
                let Some(total) = apev2_start_len(&self.buf) else {
                    if self.buf.len() < APE_FOOTER_LEN {
                        return Ok(());
                    }
                    self.start_checked = true;
                    return self.drain_tail(out).await;
                };
                if self.buf.len() < total {
                    return Ok(());
                }
                if let Some((tags, parsed_total)) = parse_apev2_tail(&self.buf[..total]) {
                    if parsed_total != total {
                        return Err(G2gError::CapsMismatch);
                    }
                    self.tags = tags;
                    self.buf.drain(..total);
                }
                self.start_checked = true;
            }
        }
        self.drain_tail(out).await
    }

    async fn drain_tail(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let Some(len) = self.buf.len().checked_sub(TAIL_HELD) else {
            return Ok(());
        };
        let data: Vec<u8> = self.buf.drain(..len).collect();
        self.emit(data, out).await
    }

    /// Split a trailing ID3v1 block off, parse and drop the APEv2 tag, then
    /// forward the payload plus the ID3v1 block (id3demux strips that).
    async fn finish(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let mut tail: Vec<u8> = core::mem::take(&mut self.buf);
        let id3v1 = split_id3v1(&mut tail);
        if let Some((tags, total)) = parse_apev2_tail(&tail) {
            if self.tags.is_empty() {
                self.tags = tags;
            }
            tail.truncate(tail.len() - total);
        }
        if !self.tags.is_empty() {
            if let Some(bus) = &self.bus {
                bus.try_post(BusMessage::Tag {
                    tags: self.tags.clone(),
                    program: None,
                });
            }
        }
        tail.extend_from_slice(&id3v1);
        self.emit(tail, out).await
    }
}

fn split_id3v1(tail: &mut Vec<u8>) -> Vec<u8> {
    let Some(start) = tail.len().checked_sub(ID3V1_LEN) else {
        return Vec::new();
    };
    if parse_id3v1(&tail[start..]).is_none() {
        return Vec::new();
    }
    tail.split_off(start)
}

impl AsyncElement for ApeDemux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

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
            "APEv2 tag reader",
            "Formatter/Metadata",
            "Reads an APEv2 tag from the tail of a stream and strips it",
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
                    self.drain(out).await?;
                }
                PipelinePacket::Eos => self.finish(out).await?,
                PipelinePacket::Flush => {
                    self.buf.clear();
                    self.tags = TagList::new();
                    self.start_checked = false;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apev2mux::ApeV2Mux;
    use crate::testutil::{data_bytes, run};
    use alloc::format;
    use g2g_core::{ByteStreamEncoding, PropValue, Tag};

    fn raw_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Raw,
        }
    }

    fn with_tags(tags: &str) -> ApeV2Mux {
        let mut mux = ApeV2Mux::new();
        mux.set_property("tags", PropValue::Str(tags.into()))
            .unwrap();
        mux
    }

    #[test]
    fn strips_the_tag_the_muxer_wrote_and_reads_its_items() {
        let payload = b"\xff\xfb\x90\x00 audio bytes";
        let title = "Sine";
        let artist = "g2g";
        let tagged = run(
            &mut with_tags(&format!("title={title},artist={artist}")),
            &raw_caps(),
            &[payload],
        );
        let file = data_bytes(&tagged.packets);
        assert!(file.len() > payload.len());

        let mut demux = ApeDemux::new();
        let stripped = run(&mut demux, &raw_caps(), &[&file]);
        assert_eq!(data_bytes(&stripped.packets), payload);
        let tags = demux.tags().tags();
        assert!(tags
            .iter()
            .any(|t| matches!(t, Tag::Title(s) if s == title)));
        assert!(tags
            .iter()
            .any(|t| matches!(t, Tag::Artist(s) if s == artist)));
    }

    #[test]
    fn a_stream_with_no_tag_passes_through() {
        let payload = b"just bytes";
        let mut demux = ApeDemux::new();
        let out = run(&mut demux, &raw_caps(), &[payload]);
        assert_eq!(data_bytes(&out.packets), payload);
        assert!(demux.tags().is_empty());
    }

    #[test]
    fn strips_a_tag_at_the_start() {
        let payload = b"audio bytes";
        let tagged = run(&mut with_tags("title=At start"), &raw_caps(), &[payload]);
        let file = data_bytes(&tagged.packets);
        let tag = &file[payload.len()..];
        let mut start_tagged = tag.to_vec();
        start_tagged.extend_from_slice(payload);

        let mut demux = ApeDemux::new();
        let stripped = run(
            &mut demux,
            &raw_caps(),
            &[&start_tagged[..12], &start_tagged[12..]],
        );
        assert_eq!(data_bytes(&stripped.packets), payload);
        assert!(demux
            .tags()
            .tags()
            .iter()
            .any(|tag| matches!(tag, Tag::Title(title) if title == "At start")));
    }
}
